use crate::{
    ffi, ffi_error,
    handle::OwnedMol,
    mols::OwnedMols,
    pool::{Cancelled, SIGNAL_EVERY, run_pool_verbose, setup_local_pool},
    salt_remover::default_salt_lines,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PySequence, PyString},
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Title of the progress bar.
const DESC: &str = "Cleaning";

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

/// Wrapper around a cleaned result that frees its molecule on drop.
///
/// This keeps an interrupted run from leaking the molecules it already cleaned. `into_cleaned`
/// takes the molecule out.
struct Done(ffi::Cleaned);

impl Done {
    fn into_cleaned(mut self) -> ffi::Cleaned {
        let cleaned = ffi::Cleaned {
            smiles: std::mem::take(&mut self.0.smiles),
            mol: self.0.mol,
        };
        self.0.mol = 0;
        cleaned
    }
}

impl Drop for Done {
    fn drop(&mut self) {
        if self.0.mol != 0 {
            ffi::free_mol_raw(self.0.mol);
        }
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
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let smiles: Vec<Option<String>> = inputs.extract()?;
    let cleaned = run_pool_verbose(py, pool, verbose, smiles.len(), DESC, true, |progress| {
        smiles
            .par_iter()
            .map_init(
                || job.cleaner().expect("salt patterns are valid"),
                |cleaner, smiles| {
                    Done(match smiles {
                        Some(s) => ffi::clean_smiles_raw(cleaner.0, s, job.want_mol),
                        None => invalid(),
                    })
                },
            )
            .map(|c| progress.track(c))
            .collect::<Result<Vec<_>, Cancelled>>()
    })?;
    to_py_list(
        py,
        cleaned.into_iter().map(Done::into_cleaned).collect(),
        job.want_mol,
    )
}

/// Cleans copies of molecules with GIL released.
fn clean_mols<'py>(
    py: Python<'py>,
    pool: &rayon::ThreadPool,
    job: &Job,
    inputs: &Bound<'py, PySequence>,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let batch =
        ffi::extract_mol_ptrs(inputs.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
    let keepalive: Bound<'py, PyAny> =
        unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };
    // private copies, made serially as copying a shared `ROMol` from several threads is not
    // known to be safe. Afterwards no thread touches the inputs, so the GIL can be released.
    let mut copies = OwnedMols(Vec::with_capacity(batch.ptrs.len()));
    for (i, &p) in batch.ptrs.iter().enumerate() {
        if i % SIGNAL_EVERY == 0 {
            py.check_signals()?;
        }
        let copy =
            ffi::clone_rwmol_raw(p).map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
        copies.0.push(copy);
    }
    drop(keepalive);
    // `clean_mol_raw` takes ownership of each copy, and copies it never got are freed on drop.
    let owned: Vec<OwnedMol> = std::mem::take(&mut copies.0)
        .into_iter()
        .map(|p| OwnedMol::from_raw(p).expect("a copy is not null"))
        .collect();
    let total = owned.len();
    let cleaned = run_pool_verbose(py, pool, verbose, total, DESC, true, |progress| {
        owned
            .into_par_iter()
            .map_init(
                || job.cleaner().expect("salt patterns are valid"),
                |cleaner, mol| Done(ffi::clean_mol_raw(cleaner.0, mol.into_raw(), !job.want_mol)),
            )
            .map(|c| progress.track(c))
            .collect::<Result<Vec<_>, Cancelled>>()
    })?;
    to_py_list(
        py,
        cleaned.into_iter().map(Done::into_cleaned).collect(),
        job.want_mol,
    )
}

/// Cleans molecules in place, with the GIL held so no Python thread uses them meanwhile.
fn clean_inplace<'py>(
    py: Python<'py>,
    pool: &rayon::ThreadPool,
    job: &Job,
    inputs: &Bound<'py, PySequence>,
    verbose: bool,
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
    let valid: Vec<bool> =
        run_pool_verbose(py, pool, verbose, distinct.len(), DESC, false, |progress| {
            distinct
                .par_iter()
                .map_init(
                    || job.cleaner().expect("salt patterns are valid"),
                    |cleaner, &p| ffi::clean_mol_inplace_raw(cleaner.0, p),
                )
                .map(|valid| progress.track(valid))
                .collect::<Result<Vec<_>, Cancelled>>()
        })?;
    drop(keepalive);
    Ok(PyList::new(py, slots.iter().map(|&s| valid[s]))?.into_any())
}

/// Tells whether inputs are SMILES or molecules, judging by the first item.
///
/// A leading `None` counts as SMILES. Returns `None` for no items.
pub(crate) fn is_smiles_input(inputs: &Bound<'_, PySequence>) -> PyResult<Option<bool>> {
    if inputs.len()? == 0 {
        return Ok(None);
    }
    let first = inputs.get_item(0)?;
    Ok(Some(first.is_none() || first.is_instance_of::<PyString>()))
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
    largest_fragment=false,
    no_salts=false,
    no_isotopes=false,
    no_stereo=false,
    isomeric=true,
    allowed_elements=None,
    neutralize=false,
    verify=true,
    inplace=false,
    n_threads=-1,
    verbose=false,
))]
#[allow(clippy::too_many_arguments)]
pub fn clean<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: Option<&str>,
    largest_fragment: bool,
    no_salts: bool,
    no_isotopes: bool,
    no_stereo: bool,
    isomeric: bool,
    allowed_elements: Option<&Bound<'py, PyAny>>,
    neutralize: bool,
    verify: bool,
    inplace: bool,
    n_threads: i32,
    verbose: bool,
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
            largest_fragment,
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

    let Some(is_smiles) = is_smiles_input(inputs)? else {
        return Ok(PyList::empty(py).into_any());
    };
    match (is_smiles, inplace) {
        (true, false) => clean_smiles(py, &pool, &job, inputs, verbose),
        (false, false) => clean_mols(py, &pool, &job, inputs, verbose),
        (false, true) => clean_inplace(py, &pool, &job, inputs, verbose),
        (true, true) => Err(PyTypeError::new_err("inplace requires molecules")),
    }
}
