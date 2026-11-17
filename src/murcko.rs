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

/// Result for one input: the scaffold as SMILES or molecule pointer, or invalid.
enum Scaffold {
    Invalid,
    Smiles(String),
    Mol(usize),
}

/// Takes a scaffold from `murcko_raw` and returns it as SMILES or keeps it as molecule.
fn finish(scaffold: usize, want_mol: bool) -> Scaffold {
    if scaffold == 0 {
        return Scaffold::Invalid;
    }
    if want_mol {
        return Scaffold::Mol(scaffold);
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
    let smiles = ffi::mol_to_smiles_raw(scaffold, &opts);
    ffi::free_mol_raw(scaffold);
    smiles.map_or(Scaffold::Invalid, Scaffold::Smiles)
}

/// Returns the scaffold of SMILES, which is invalid for `None`, an empty string, or a parse error.
fn from_smiles(smiles: Option<&String>, want_mol: bool) -> Scaffold {
    let Some(smiles) = smiles.filter(|s| !s.is_empty()) else {
        return Scaffold::Invalid;
    };
    let mol = ffi::parse_smiles_raw(smiles, true);
    if mol == 0 {
        return Scaffold::Invalid;
    }
    let scaffold = ffi::murcko_raw(mol);
    ffi::free_mol_raw(mol);
    finish(scaffold, want_mol)
}

/// Computes Murcko scaffolds of SMILES or molecules on real threads.
///
/// Molecules are copied first, serially, so the inputs stay unchanged. Releases the GIL for the
/// computation. A molecule without rings has an empty scaffold, which is not invalid.
#[pyfunction]
#[pyo3(signature = (inputs, output="smiles", *, n_threads=-1))]
pub fn murcko_scaffolds<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
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

    let scaffolds: Vec<Scaffold> = if is_smiles {
        let smiles: Vec<Option<String>> = inputs.extract()?;
        run_pool(py, &pool, || {
            smiles
                .par_iter()
                .map(|s| from_smiles(s.as_ref(), want_mol))
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
        let owned = std::mem::take(&mut copies.0);
        run_pool(py, &pool, || {
            owned
                .par_iter()
                .map(|&copy| {
                    let scaffold = ffi::murcko_raw(copy);
                    ffi::free_mol_raw(copy);
                    finish(scaffold, want_mol)
                })
                .collect()
        })
    };

    if !want_mol {
        let smiles = scaffolds.into_iter().map(|s| match s {
            Scaffold::Smiles(smiles) => Some(smiles),
            _ => None,
        });
        return Ok(PyList::new(py, smiles)?.into_any());
    }
    let ptrs: Vec<usize> = scaffolds
        .iter()
        .map(|s| match s {
            Scaffold::Mol(ptr) => *ptr,
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
