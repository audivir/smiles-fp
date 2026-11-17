use crate::{
    clean::is_smiles_input,
    ffi, ffi_error,
    mols::OwnedMols,
    pool::{run_pool, setup_local_pool},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PySequence},
};
use rayon::prelude::*;

/// Result for one molecule: SMILES or a molecule pointer, or invalid.
pub(crate) enum Output {
    Invalid,
    Smiles(String),
    Mol(usize),
}

/// Reads the `output` argument, which is `true` for molecules.
pub(crate) fn want_mol(output: &str) -> PyResult<bool> {
    match output {
        "smiles" => Ok(false),
        "mol" => Ok(true),
        _ => Err(PyValueError::new_err(format!(
            "Unknown output '{output}', expected 'smiles' or 'mol'"
        ))),
    }
}

/// Tells whether SMILES parse to a molecule with atoms.
fn reparses(smiles: &str) -> bool {
    let mol = ffi::parse_smiles_raw(smiles, true);
    if mol == 0 {
        return false;
    }
    let has_atoms = ffi::mol_num_atoms_raw(mol) > 0;
    ffi::free_mol_raw(mol);
    has_atoms
}

/// Returns a new molecule as SMILES, or keeps it as molecule.
///
/// With `verify`, a result whose SMILES do not parse to a molecule with atoms is invalid, whether
/// it is returned as SMILES or as molecule.
pub(crate) fn finish(mol: usize, want_mol: bool, verify: bool) -> Output {
    if mol == 0 {
        return Output::Invalid;
    }
    // the options of `Chem.MolToSmiles`.
    let opts = ffi::SmilesOpts {
        isomeric_smiles: true,
        kekule_smiles: false,
        rooted_at_atom: -1,
        canonical: true,
        all_bonds_explicit: false,
        all_hs_explicit: false,
        ignore_atom_map_numbers: false,
    };
    let smiles = ffi::mol_to_smiles_raw(mol, &opts);
    // molecules are checked through their SMILES too, so that both outputs agree on what is valid.
    let valid = !verify || smiles.as_ref().is_ok_and(|s| reparses(s));
    if want_mol {
        if valid {
            return Output::Mol(mol);
        }
        ffi::free_mol_raw(mol);
        return Output::Invalid;
    }
    ffi::free_mol_raw(mol);
    match smiles {
        Ok(s) if valid => Output::Smiles(s),
        _ => Output::Invalid,
    }
}

/// Wraps molecule pointers in a Python list, with 0 becoming `None`.
pub(crate) fn wrap_ptrs<'py>(py: Python<'py>, ptrs: &[usize]) -> PyResult<Bound<'py, PyList>> {
    let wrapped = ffi::wrap_mols(ptrs);
    if wrapped == 0 {
        return Err(PyErr::take(py)
            .unwrap_or_else(|| PyRuntimeError::new_err("Failed to wrap molecules in C++")));
    }
    Ok(
        unsafe { Bound::<PyAny>::from_owned_ptr(py, wrapped as *mut pyo3::ffi::PyObject) }
            .cast_into()?,
    )
}

/// Turns results into a Python list, with `None` for invalid ones.
pub(crate) fn outputs_to_py<'py>(
    py: Python<'py>,
    results: Vec<Output>,
    want_mol: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !want_mol {
        let smiles = results.into_iter().map(|r| match r {
            Output::Smiles(s) => Some(s),
            _ => None,
        });
        return Ok(PyList::new(py, smiles)?.into_any());
    }
    let ptrs: Vec<usize> = results
        .iter()
        .map(|r| match r {
            Output::Mol(ptr) => *ptr,
            _ => 0,
        })
        .collect();
    Ok(wrap_ptrs(py, &ptrs)?.into_any())
}

/// Inputs of a batch: SMILES or private copies of molecules.
pub(crate) enum Inputs {
    Smiles(Vec<Option<String>>),
    Mols(OwnedMols),
}

/// Copies molecules to private ones, serially.
///
/// Copying a shared `ROMol` from several threads is not known to be safe. Afterwards no thread
/// touches the originals.
pub(crate) fn copy_mols(py: Python<'_>, seq: &Bound<'_, PySequence>) -> PyResult<OwnedMols> {
    let batch =
        ffi::extract_mol_ptrs(seq.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
    let keepalive: Bound<'_, PyAny> =
        unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };
    let mut copies = OwnedMols(Vec::with_capacity(batch.ptrs.len()));
    for &p in &batch.ptrs {
        let copy =
            ffi::clone_mol_raw(p).map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
        copies.0.push(copy);
    }
    drop(keepalive);
    Ok(copies)
}

/// Reads SMILES, where `None` and empty strings are invalid later, or molecules.
///
/// Returns `None` for no items.
pub(crate) fn read_inputs(
    py: Python<'_>,
    inputs: &Bound<'_, PySequence>,
) -> PyResult<Option<Inputs>> {
    Ok(match is_smiles_input(inputs)? {
        None => None,
        Some(true) => Some(Inputs::Smiles(inputs.extract()?)),
        Some(false) => Some(Inputs::Mols(copy_mols(py, inputs)?)),
    })
}

/// Parses SMILES into a molecule for `f`, which is `None` for invalid SMILES.
fn parse_for<R>(smiles: &Option<String>, f: impl FnOnce(usize) -> R) -> Option<R> {
    let smiles = smiles.as_ref().filter(|s| !s.is_empty())?;
    let mol = ffi::parse_smiles_raw(smiles, true);
    if mol == 0 {
        return None;
    }
    let result = f(mol);
    ffi::free_mol_raw(mol);
    Some(result)
}

/// Applies `f` to every input molecule on real threads, with `None` for invalid SMILES.
///
/// Each thread gets its own state from `init`, for objects that cannot be shared. `f` must not free
/// or keep the molecule it gets. Releases the GIL for the computation.
pub(crate) fn map_inputs<S, R, I, F>(
    py: Python<'_>,
    inputs: &Inputs,
    n_threads: i32,
    init: I,
    f: F,
) -> PyResult<Vec<Option<R>>>
where
    R: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> R + Sync + Send,
{
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    Ok(run_pool(py, &pool, || match inputs {
        Inputs::Smiles(smiles) => smiles
            .par_iter()
            .map_init(&init, |state, s| parse_for(s, |mol| f(state, mol)))
            .collect(),
        Inputs::Mols(copies) => copies
            .0
            .par_iter()
            .map_init(&init, |state, &mol| Some(f(state, mol)))
            .collect(),
    }))
}

/// Applies `op`, which returns a new owned molecule or 0, to every input molecule.
///
/// Returns a list of SMILES or molecules, with `None` for invalid inputs.
pub(crate) fn transform<'py, S, I, F>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    n_threads: i32,
    verify: bool,
    init: I,
    op: F,
) -> PyResult<Bound<'py, PyAny>>
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> usize + Sync + Send,
{
    let want_mol = want_mol(output)?;
    let Some(inputs) = read_inputs(py, inputs)? else {
        return Ok(PyList::empty(py).into_any());
    };
    let results = map_inputs(py, &inputs, n_threads, init, |state, mol| {
        finish(op(state, mol), want_mol, verify)
    })?;
    outputs_to_py(
        py,
        results
            .into_iter()
            .map(|r| r.unwrap_or(Output::Invalid))
            .collect(),
        want_mol,
    )
}
