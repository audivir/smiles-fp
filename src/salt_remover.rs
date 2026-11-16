use crate::{OwnedMols, ffi, ffi_error, setup_local_pool};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyAny, PySequence},
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Default salt patterns, a copy of `Salts.txt` of RDKit with normalized whitespace.
const DEFAULT_SALTS: &str = include_str!("salts.txt");

/// Returns the default SMARTS, one per line, read like `SaltRemover`: text before `//`, first word.
pub(crate) fn default_salt_lines() -> String {
    let text = DEFAULT_SALTS;
    text.lines()
        .filter_map(|l| l.split("//").next()?.split_whitespace().next())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Wrapper around the query molecules of the default salts, owned by one thread.
struct SaltStripper(usize);

impl SaltStripper {
    fn new() -> Result<Self, String> {
        ffi::make_salt_stripper(&default_salt_lines())
            .map(Self)
            .map_err(|e| e.what().to_string())
    }

    /// Returns a stripped copy of the molecule, or 0 if sanitizing it fails.
    fn strip(&self, mol: usize, keep_last: bool) -> usize {
        ffi::strip_salts_raw(self.0, mol, keep_last)
    }
}

impl Drop for SaltStripper {
    fn drop(&mut self) {
        ffi::free_salt_stripper(self.0);
    }
}

/// Removes the default salts from copies of molecules like `SaltRemover().StripMol`.
///
/// Returns `None` for a molecule whose stripped copy fails to sanitize. Holds the GIL.
#[pyfunction]
#[pyo3(name = "_strip_salts", signature = (mols, dont_remove_everything=false, *, n_threads=-1))]
pub fn strip_salts<'py>(
    py: Python<'py>,
    mols: &Bound<'py, PySequence>,
    dont_remove_everything: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    // fails early on an invalid pattern, so the per-thread strippers below cannot.
    SaltStripper::new().map_err(PyRuntimeError::new_err)?;
    let batch =
        ffi::extract_mol_ptrs(mols.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
    let keepalive: Bound<'py, PyAny> =
        unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };

    // a molecule that appears several times is stripped once, as threads must not share one.
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
    let mut stripped = OwnedMols(pool.install(|| {
        distinct
            .par_iter()
            .map_init(
                || SaltStripper::new().expect("salt patterns are valid"),
                |stripper, &p| stripper.strip(p, dont_remove_everything),
            )
            .collect()
    }));
    drop(keepalive);

    let mut out = OwnedMols(Vec::with_capacity(slots.len()));
    let mut first_pos: Vec<Option<usize>> = vec![None; distinct.len()];
    for (i, &s) in slots.iter().enumerate() {
        out.0.push(match first_pos[s] {
            None => {
                first_pos[s] = Some(i);
                std::mem::take(&mut stripped.0[s])
            }
            Some(_) if out.0[first_pos[s].unwrap()] == 0 => 0,
            Some(j) => ffi::clone_mol_raw(out.0[j])
                .map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?,
        });
    }
    // `wrap_mols` takes ownership of every pointer, also when it fails.
    let wrapped = ffi::wrap_mols(&std::mem::take(&mut out.0));
    if wrapped == 0 {
        return Err(PyErr::take(py)
            .unwrap_or_else(|| PyRuntimeError::new_err("Failed to wrap molecules in C++")));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, wrapped as *mut pyo3::ffi::PyObject) })
}
