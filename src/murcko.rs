use crate::{ffi, handle::OwnedMol, transform::transform};
use pyo3::{
    prelude::*,
    types::{PyAny, PySequence},
};

/// Computes Murcko scaffolds of SMILES or molecules on real threads.
///
/// Molecules are copied first, serially, so the inputs stay unchanged. Releases the GIL for the
/// computation. A molecule without rings has an empty scaffold, which is not invalid.
#[pyfunction]
#[pyo3(signature = (
    inputs, output="smiles", *, generic=false, include_chirality=true, n_threads=-1, verbose=false,
))]
pub fn murcko_scaffolds<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    generic: bool,
    include_chirality: bool,
    n_threads: i32,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    transform(
        py,
        inputs,
        output,
        n_threads,
        verbose,
        "Murcko scaffolds",
        false,
        || (),
        |_, mol| OwnedMol::from_raw(ffi::murcko_raw(mol.as_raw(), generic, include_chirality)),
    )
}
