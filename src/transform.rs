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

/// Result for one input: a molecule pointer, or 0 if invalid.
enum Output {
    Invalid,
    Smiles(String),
    Mol(usize),
}

/// Takes a new molecule from a transformation and returns it as SMILES or keeps it as molecule.
///
/// With `verify`, molecules without atoms and SMILES that do not parse again are invalid.
fn finish(mol: usize, want_mol: bool, verify: bool) -> Output {
    if mol == 0 {
        return Output::Invalid;
    }
    if want_mol {
        if verify && ffi::mol_num_atoms_raw(mol) == 0 {
            ffi::free_mol_raw(mol);
            return Output::Invalid;
        }
        return Output::Mol(mol);
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
    ffi::free_mol_raw(mol);
    match smiles {
        Ok(s) if !verify || reparses(&s) => Output::Smiles(s),
        _ => Output::Invalid,
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

/// Applies `op` to every input molecule on real threads.
///
/// `op` takes a molecule pointer and returns a new owned molecule or 0. Each thread gets its own
/// state from `init`, for objects that cannot be shared. Inputs are SMILES, where `None` and empty
/// strings are invalid, or molecules, which are copied serially first and stay unchanged. Releases
/// the GIL for the computation.
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
    let want_mol = match output {
        "smiles" => false,
        "mol" => true,
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown output '{output}', expected 'smiles' or 'mol'"
            )));
        }
    };
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let Some(is_smiles) = is_smiles_input(inputs)? else {
        return Ok(PyList::empty(py).into_any());
    };

    let results: Vec<Output> = if is_smiles {
        let smiles: Vec<Option<String>> = inputs.extract()?;
        run_pool(py, &pool, || {
            smiles
                .par_iter()
                .map_init(&init, |state, s| {
                    let Some(s) = s.as_ref().filter(|s| !s.is_empty()) else {
                        return Output::Invalid;
                    };
                    let mol = ffi::parse_smiles_raw(s, true);
                    if mol == 0 {
                        return Output::Invalid;
                    }
                    let new = op(state, mol);
                    ffi::free_mol_raw(mol);
                    finish(new, want_mol, verify)
                })
                .collect()
        })
    } else {
        let batch =
            ffi::extract_mol_ptrs(inputs.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
        let keepalive: Bound<'py, PyAny> =
            unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };
        // private copies, made serially as copying a shared `ROMol` from several threads is not
        // known to be safe. Afterwards no thread touches the inputs, so the GIL can be released.
        let mut copies = OwnedMols(Vec::with_capacity(batch.ptrs.len()));
        for &p in &batch.ptrs {
            let copy =
                ffi::clone_mol_raw(p).map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
            copies.0.push(copy);
        }
        drop(keepalive);
        run_pool(py, &pool, || {
            copies
                .0
                .par_iter()
                .map_init(&init, |state, &copy| {
                    finish(op(state, copy), want_mol, verify)
                })
                .collect()
        })
    };

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
    let wrapped = ffi::wrap_mols(&ptrs);
    if wrapped == 0 {
        return Err(PyErr::take(py)
            .unwrap_or_else(|| PyRuntimeError::new_err("Failed to wrap molecules in C++")));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, wrapped as *mut pyo3::ffi::PyObject) })
}
