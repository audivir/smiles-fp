use crate::{ffi, transform::transform};
use pyo3::{
    prelude::*,
    types::{PyAny, PySequence},
};

/// Owner of the tautomer enumerator of one thread, which is freed on drop.
struct Tautomerizer(usize);

impl Drop for Tautomerizer {
    fn drop(&mut self) {
        ffi::free_tautomerizer(self.0);
    }
}

/// Computes canonical tautomers of SMILES or molecules on real threads.
///
/// Each thread builds its own enumerator. Molecules are copied first, serially, so the inputs stay
/// unchanged. Releases the GIL for the computation.
#[pyfunction]
#[pyo3(signature = (inputs, output="smiles", *, max_tautomers=10_000, max_transforms=10_000, n_threads=-1))]
pub fn canonical_tautomers<'py>(
    py: Python<'py>,
    inputs: &Bound<'py, PySequence>,
    output: &str,
    max_tautomers: u32,
    max_transforms: u32,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    transform(
        py,
        inputs,
        output,
        n_threads,
        true,
        || Tautomerizer(ffi::make_tautomerizer(max_tautomers, max_transforms)),
        |t, mol| ffi::tautomer_raw(t.0, mol),
    )
}
