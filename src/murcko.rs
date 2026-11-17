use crate::{ffi, transform::transform};
use pyo3::{
    prelude::*,
    types::{PyAny, PySequence},
};

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
    transform(
        py,
        inputs,
        output,
        n_threads,
        false,
        || (),
        |_, mol| ffi::murcko_raw(mol),
    )
}
