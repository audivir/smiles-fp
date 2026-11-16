use pyo3::prelude::*;

/// Thread-safe pointer wrapper
#[derive(Copy, Clone)]
pub(crate) struct SyncPtr<T>(pub(crate) *mut T);

unsafe impl<T> Send for SyncPtr<T> {}

unsafe impl<T> Sync for SyncPtr<T> {}

/// Get the actual number of threads
#[inline(always)]
pub(crate) fn get_n_threads(n_threads: i32) -> usize {
    if n_threads > 0 {
        n_threads as usize
    } else {
        rayon::current_num_threads()
    }
}

/// Set up a local Rayon thread pool
#[inline(always)]
pub(crate) fn setup_local_pool(n_threads: i32) -> Result<rayon::ThreadPool, String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(get_n_threads(n_threads))
        .build()
        .map_err(|e| format!("Failed to build thread pool: {}", e))
}

/// Runs `f` on Rayon pool with GIL released, so `f` must not touch Python.
pub(crate) fn run_pool<R: Send, F: FnOnce() -> R + Send>(
    py: Python<'_>,
    pool: &rayon::ThreadPool,
    f: F,
) -> R {
    py.detach(|| pool.install(f))
}
