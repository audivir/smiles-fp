use crate::{
    ffi, ffi_error,
    mols::OwnedMols,
    pool::{run_pool, setup_local_pool},
    salt_remover::default_salt_lines,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PySequence, PyString},
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Wrapper around the C++ cleaner, which one thread owns.
struct Cleaner(usize);

impl Drop for Cleaner {
    fn drop(&mut self) {
        ffi::free_cleaner(self.0);
    }
}

/// Container for the settings that every thread builds its own cleaner from.
struct Job {
    opts: ffi::CleanOpts,
    allowed: Vec<u32>,
    want_mol: bool,
}

impl Job {
    fn cleaner(&self) -> Result<Cleaner, String> {
        ffi::make_cleaner(&self.opts, &self.allowed, default_salt_lines())
            .map(Cleaner)
            .map_err(|e| e.what().to_string())
    }
}

/// Returns a result for an invalid input.
fn invalid() -> ffi::Cleaned {
    ffi::Cleaned {
        smiles: String::new(),
        mol: 0,
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

/// Cleans SMILES with GIL released.
fn clean_smiles<'py>(
    py: Python<'py>,
    pool: &rayon::ThreadPool,
    job: &Job,
    inputs: &Bound<'py, PySequence>,
) -> PyResult<Bound<'py, PyAny>> {
    let smiles: Vec<Option<String>> = inputs.extract()?;
    let cleaned = run_pool(py, pool, || {
        smiles
            .par_iter()
            .map_init(
                || job.cleaner().expect("salt patterns are valid"),
                |cleaner, smiles| match smiles {
                    Some(s) => ffi::clean_smiles_raw(cleaner.0, s, job.want_mol),
                    None => invalid(),
                },
            )
            .collect()
    });
    to_py_list(py, cleaned, job.want_mol)
}

/// Cleans copies of molecules with GIL released.
fn clean_mols<'py>(
    py: Python<'py>,
    pool: &rayon::ThreadPool,
    job: &Job,
    inputs: &Bound<'py, PySequence>,
) -> PyResult<Bound<'py, PyAny>> {
    let batch =
        ffi::extract_mol_ptrs(inputs.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
    let keepalive: Bound<'py, PyAny> =
        unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };
    // private copies, made serially as copying a shared `ROMol` from several threads is not
    // known to be safe. Afterwards no thread touches the inputs, so the GIL can be released.
    let mut copies = OwnedMols(Vec::with_capacity(batch.ptrs.len()));
    for &p in &batch.ptrs {
        let copy =
            ffi::clone_rwmol_raw(p).map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
        copies.0.push(copy);
    }
    drop(keepalive);
    // `clean_mol_raw` takes ownership of each copy.
    let owned = std::mem::take(&mut copies.0);
    let cleaned = run_pool(py, pool, || {
        owned
            .par_iter()
            .map_init(
                || job.cleaner().expect("salt patterns are valid"),
                |cleaner, &p| ffi::clean_mol_raw(cleaner.0, p, !job.want_mol),
            )
            .collect()
    });
    to_py_list(py, cleaned, job.want_mol)
}

/// Cleans molecules in place, with the GIL held so no Python thread uses them meanwhile.
fn clean_inplace<'py>(
    py: Python<'py>,
    pool: &rayon::ThreadPool,
    job: &Job,
    inputs: &Bound<'py, PySequence>,
) -> PyResult<Bound<'py, PyAny>> {
    let batch =
        ffi::extract_mol_ptrs(inputs.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
    let keepalive: Bound<'py, PyAny> =
        unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };
    // a molecule that appears several times is cleaned once, as threads must not share one.
    let mut distinct: Vec<usize> = Vec::new();
    let mut slot_of: HashMap<usize, usize> = HashMap::with_capacity(batch.ptrs.len());
    let slots: Vec<usize> = batch
        .ptrs
        .iter()
        .map(|&p| {
            *slot_of.entry(p).or_insert_with(|| {
                distinct.push(p);
                distinct.len() - 1
            })
        })
        .collect();
    let valid: Vec<bool> = pool.install(|| {
        distinct
            .par_iter()
            .map_init(
                || job.cleaner().expect("salt patterns are valid"),
                |cleaner, &p| ffi::clean_mol_inplace_raw(cleaner.0, p),
            )
            .collect()
    });
    drop(keepalive);
    Ok(PyList::new(py, slots.iter().map(|&s| valid[s]))?.into_any())
}

/// Cleans SMILES or molecules on real threads.
///
/// SMILES and copies of molecules are cleaned with the GIL released, molecules in place with the
/// GIL held. Returns canonical SMILES or molecules, with `None` for invalid inputs, or with
/// `inplace` whether each molecule is valid. For SMILES the molecules are parsed from the written
/// SMILES, for molecules they are the cleaned copies.
#[pyfunction]
#[pyo3(signature = (
    inputs,
    output=None,
    *,
    no_salts=false,
    no_isotopes=false,
    no_stereo=false,
    isomeric=true,
    allowed_elements=None,
    neutralize=false,
    verify=true,
    inplace=false,
    n_threads=-1,
))]
#[allow(clippy::too_many_arguments)]
pub fn clean<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: Option<&str>,
    no_salts: bool,
    no_isotopes: bool,
    no_stereo: bool,
    isomeric: bool,
    allowed_elements: Option<&Bound<'py, PyAny>>,
    neutralize: bool,
    verify: bool,
    inplace: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    // the default is "smiles", or "mol" with `inplace`.
    let want_mol = match output.unwrap_or(if inplace { "mol" } else { "smiles" }) {
        "smiles" => false,
        "mol" => true,
        output => {
            return Err(PyValueError::new_err(format!(
                "Unknown output '{output}', expected 'smiles' or 'mol'"
            )));
        }
    };
    if inplace && !want_mol {
        return Err(PyValueError::new_err("inplace requires output 'mol'"));
    }
    let allowed: Vec<u32> = match allowed_elements.filter(|a| !a.is_none()) {
        Some(a) => a
            .try_iter()?
            .map(|z| z?.extract())
            .collect::<PyResult<_>>()?,
        None => Vec::new(),
    };
    let job = Job {
        opts: ffi::CleanOpts {
            no_salts,
            no_isotopes,
            no_stereo,
            isomeric,
            neutralize,
            verify,
        },
        allowed,
        want_mol,
    };
    // fails early on an invalid pattern, so the per-thread cleaners cannot.
    job.cleaner().map_err(PyRuntimeError::new_err)?;
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;

    let is_smiles = match inputs.len()? {
        0 => return Ok(PyList::empty(py).into_any()),
        _ => {
            let first = inputs.get_item(0)?;
            first.is_none() || first.is_instance_of::<PyString>()
        }
    };
    match (is_smiles, inplace) {
        (true, false) => clean_smiles(py, &pool, &job, inputs),
        (false, false) => clean_mols(py, &pool, &job, inputs),
        (false, true) => clean_inplace(py, &pool, &job, inputs),
        (true, true) => Err(PyTypeError::new_err("inplace requires molecules")),
    }
}
