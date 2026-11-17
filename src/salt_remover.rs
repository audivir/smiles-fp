use crate::{ffi, ffi_error, handle::OwnedMol, pool::setup_local_pool};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyAny, PySequence},
};
use rayon::prelude::*;
use std::{collections::HashMap, sync::LazyLock};

/// Default salt patterns, a copy of `Salts.txt` of RDKit with normalized whitespace.
const DEFAULT_SALTS: &str = include_str!("salts.txt");

/// Default SMARTS, one per line, read like `SaltRemover`: text before `//`, first word.
static SALT_LINES: LazyLock<String> = LazyLock::new(|| {
    DEFAULT_SALTS
        .lines()
        .filter_map(|l| l.split("//").next()?.split_whitespace().next())
        .collect::<Vec<_>>()
        .join("\n")
});

pub(crate) fn default_salt_lines() -> &'static str {
    &SALT_LINES
}

/// Wrapper around the query molecules of the default salts, owned by one thread.
struct SaltStripper(usize);

impl SaltStripper {
    fn new() -> Result<Self, String> {
        ffi::make_salt_stripper(default_salt_lines())
            .map(Self)
            .map_err(|e| e.what().to_string())
    }

    /// Returns a stripped copy of the molecule, or `None` if sanitizing it fails.
    fn strip(&self, mol: usize, keep_last: bool) -> Option<OwnedMol> {
        OwnedMol::from_raw(ffi::strip_salts_raw(self.0, mol, keep_last))
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
    let mut stripped: Vec<Option<OwnedMol>> = pool.install(|| {
        distinct
            .par_iter()
            .map_init(
                || SaltStripper::new().expect("salt patterns are valid"),
                |stripper, &p| stripper.strip(p, dont_remove_everything),
            )
            .collect()
    });
    drop(keepalive);

    let mut out: Vec<Option<OwnedMol>> = Vec::with_capacity(slots.len());
    let mut first_pos: Vec<Option<usize>> = vec![None; distinct.len()];
    for (i, &s) in slots.iter().enumerate() {
        let item = match first_pos[s] {
            None => {
                first_pos[s] = Some(i);
                stripped[s].take()
            }
            Some(j) => match &out[j] {
                None => None,
                Some(first) => {
                    let copy = ffi::clone_mol_raw(first.as_raw())
                        .map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
                    OwnedMol::from_raw(copy)
                }
            },
        };
        out.push(item);
    }
    // `wrap_mols` takes ownership of every pointer, also when it fails.
    let ptrs: Vec<usize> = out
        .into_iter()
        .map(|m| m.map_or(0, OwnedMol::into_raw))
        .collect();
    let wrapped = ffi::wrap_mols(&ptrs);
    if wrapped == 0 {
        return Err(PyErr::take(py)
            .unwrap_or_else(|| PyRuntimeError::new_err("Failed to wrap molecules in C++")));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, wrapped as *mut pyo3::ffi::PyObject) })
}
