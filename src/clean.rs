use crate::{
    ffi, ffi_error,
    mols::OwnedMols,
    pool::{run_pool, setup_local_pool},
    salt_remover::default_salt_lines,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PySequence, PyString},
};
use rayon::prelude::*;

/// Wrapper around the C++ cleaner, which one thread owns.
struct Cleaner(usize);

impl Cleaner {
    fn new(opts: &ffi::CleanOpts, allowed: &[u32]) -> Result<Self, String> {
        ffi::make_cleaner(opts, allowed, &default_salt_lines())
            .map(Self)
            .map_err(|e| e.what().to_string())
    }
}

impl Drop for Cleaner {
    fn drop(&mut self) {
        ffi::free_cleaner(self.0);
    }
}

/// Wraps the results as a list of SMILES or of molecules, with `None` for invalid inputs.
fn to_py_list<'py>(
    py: Python<'py>,
    cleaned: Vec<ffi::Cleaned>,
    want_mol: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !want_mol {
        let smiles = cleaned
            .into_iter()
            .map(|c| (!c.smiles.is_empty()).then_some(c.smiles));
        return Ok(PyList::new(py, smiles)?.into_any());
    }
    let ptrs: Vec<usize> = cleaned.iter().map(|c| c.mol).collect();
    let wrapped = ffi::wrap_mols(&ptrs);
    if wrapped == 0 {
        return Err(PyErr::take(py)
            .unwrap_or_else(|| PyRuntimeError::new_err("Failed to wrap molecules in C++")));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, wrapped as *mut pyo3::ffi::PyObject) })
}

/// Cleans SMILES or molecules on real threads with GIL released.
///
/// Molecules are copied first, serially, so the inputs stay unchanged. Returns canonical SMILES
/// or molecules, with `None` for invalid inputs. For SMILES the molecules are parsed from the
/// written SMILES, for molecules they are the cleaned copies.
#[pyfunction]
#[pyo3(signature = (
    inputs,
    output="smiles",
    *,
    no_salts=false,
    no_isotopes=false,
    no_stereo=false,
    isomeric=true,
    allowed_elements=None,
    neutralize=false,
    n_threads=-1,
))]
#[allow(clippy::too_many_arguments)]
pub fn clean<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    no_salts: bool,
    no_isotopes: bool,
    no_stereo: bool,
    isomeric: bool,
    allowed_elements: Option<&Bound<'py, PyAny>>,
    neutralize: bool,
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
    let opts = ffi::CleanOpts {
        no_salts,
        no_isotopes,
        no_stereo,
        isomeric,
        neutralize,
    };
    let allowed: Vec<u32> = match allowed_elements.filter(|a| !a.is_none()) {
        Some(a) => a
            .try_iter()?
            .map(|z| z?.extract())
            .collect::<PyResult<_>>()?,
        None => Vec::new(),
    };
    // fails early on an invalid pattern, so the per-thread cleaners below cannot.
    Cleaner::new(&opts, &allowed).map_err(PyRuntimeError::new_err)?;
    let new_cleaner = || Cleaner::new(&opts, &allowed).expect("salt patterns are valid");

    let is_smiles = match inputs.len()? {
        0 => true,
        _ => {
            let first = inputs.get_item(0)?;
            first.is_none() || first.is_instance_of::<PyString>()
        }
    };
    let cleaned: Vec<ffi::Cleaned> = if is_smiles {
        let smiles: Vec<Option<String>> = inputs.extract()?;
        run_pool(py, &pool, || {
            smiles
                .par_iter()
                .map_init(new_cleaner, |cleaner, smiles| match smiles {
                    Some(s) => ffi::clean_smiles_raw(cleaner.0, s, want_mol),
                    None => ffi::Cleaned {
                        smiles: String::new(),
                        mol: 0,
                    },
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
            let copy = ffi::clone_rwmol_raw(p)
                .map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
            copies.0.push(copy);
        }
        drop(keepalive);
        // `clean_mol_raw` takes ownership of each copy.
        let owned = std::mem::take(&mut copies.0);
        run_pool(py, &pool, || {
            owned
                .par_iter()
                .map_init(new_cleaner, |cleaner, &p| {
                    ffi::clean_mol_raw(cleaner.0, p, !want_mol)
                })
                .collect()
        })
    };
    to_py_list(py, cleaned, want_mol)
}
