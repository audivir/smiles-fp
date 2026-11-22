use pyo3::{exceptions::PyKeyboardInterrupt, prelude::*, types::PyDict};
use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

/// Interval at which the progress bar is refreshed and Python signals are checked.
const POLL: Duration = Duration::from_millis(100);

/// Number of items from which a call is watched for `KeyboardInterrupt` without `verbose`. Shorter
/// calls finish too fast to matter and skip the watching thread.
const WATCH_FROM: usize = 1000;

/// Number of items after which a serial loop on the calling thread checks for Python signals.
pub(crate) const SIGNAL_EVERY: usize = 1024;

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

/// Error for work that stopped early after Python was interrupted.
pub(crate) struct Cancelled;

/// Container for the count of finished items and the stop flag of one call.
///
/// Workers pass each finished item through `track`, which counts it for the progress bar. After a
/// cancel, `track` returns `Err(Cancelled)` instead, so collecting into a `Result` stops Rayon from
/// starting more items.
#[derive(Default)]
pub(crate) struct Progress {
    done: AtomicUsize,
    cancelled: AtomicBool,
}

impl Progress {
    /// Counts one finished item, or returns `Err` after a cancel and drops the item.
    #[inline(always)]
    pub(crate) fn track<T>(&self, item: T) -> Result<T, Cancelled> {
        self.track_n(item, 1)
    }

    /// Counts `n` finished items at once, or returns `Err` after a cancel.
    #[inline(always)]
    pub(crate) fn track_n<T>(&self, item: T, n: usize) -> Result<T, Cancelled> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err(Cancelled);
        }
        self.done.fetch_add(n, Ordering::Relaxed);
        Ok(item)
    }

    /// Returns the number of finished items.
    fn done(&self) -> usize {
        self.done.load(Ordering::Relaxed)
    }

    /// Makes `track` fail from now on.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// Wrapper around a `tqdm` bar that remembers how many items it already shows.
struct Bar<'py> {
    bar: Bound<'py, PyAny>,
    shown: usize,
}

impl<'py> Bar<'py> {
    /// Opens a bar titled `desc` for `total` items.
    fn open(py: Python<'py>, total: usize, desc: &str) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("total", total)?;
        kwargs.set_item("desc", desc)?;
        kwargs.set_item("unit", "mol")?;
        let bar = py
            .import("tqdm.auto")?
            .getattr("tqdm")?
            .call((), Some(&kwargs))?;
        Ok(Self { bar, shown: 0 })
    }

    /// Advances the bar to `done` items.
    ///
    /// This runs Python code, where a pending Ctrl-C is raised.
    fn advance_to(&mut self, done: usize) -> PyResult<()> {
        if done > self.shown {
            self.bar.call_method1("update", (done - self.shown,))?;
            self.shown = done;
        }
        Ok(())
    }

    /// Closes the bar and leaves it at the point it reached.
    fn close(&self) {
        let _ = self.bar.call_method0("close");
    }
}

/// Container for what a watched run produced.
struct Watched<R> {
    /// Result of the workers, `Err` if they were cancelled.
    result: Result<R, Cancelled>,
    /// First Python error from the bar or a signal handler, which cancelled the workers.
    error: Option<PyErr>,
}

/// Redraws the bar and checks for Python signals on the calling thread.
fn poll(py: Python<'_>, progress: &Progress, bar: Option<&mut Bar<'_>>) -> PyResult<()> {
    if let Some(bar) = bar {
        bar.advance_to(progress.done())?;
    }
    py.check_signals()
}

/// Runs `f` on a helper thread while the calling thread polls every `POLL`.
///
/// With `release_gil` the calling thread waits without the GIL and takes it only to poll.
fn watch<R: Send, F: FnOnce(&Progress) -> Result<R, Cancelled> + Send>(
    py: Python<'_>,
    pool: &rayon::ThreadPool,
    release_gil: bool,
    progress: &Progress,
    bar: &mut Option<Bar<'_>>,
    f: F,
) -> Watched<R> {
    let (tx, rx) = mpsc::channel();
    // `Receiver` is not `Sync`, which waiting without the GIL needs.
    let rx = Mutex::new(rx);
    let mut error = None;
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            let _ = tx.send(pool.install(|| f(progress)));
        });
        loop {
            let wait = || rx.lock().expect("receiver lock").recv_timeout(POLL);
            let received = if release_gil { py.detach(wait) } else { wait() };
            // the first error cancels the workers, and polling stops so it is raised only once.
            if error.is_none()
                && let Err(e) = poll(py, progress, bar.as_mut())
            {
                progress.cancel();
                error = Some(e);
            }
            match received {
                Ok(result) => break result,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                // the helper thread panicked, which `scope` raises once it joins.
                Err(mpsc::RecvTimeoutError::Disconnected) => break Err(Cancelled),
            }
        }
    });
    Watched { result, error }
}

/// Builds the error that Python raises for Ctrl-C.
fn interrupted() -> PyErr {
    PyKeyboardInterrupt::new_err(())
}

/// Runs `f` on Rayon pool with an optional `tqdm` bar, stoppable by Ctrl-C.
///
/// `f` passes each finished item through `Progress::track`, with `total` items expected. With
/// `release_gil` the GIL is released meanwhile and `f` must not touch Python. Otherwise the GIL
/// stays held, so no other Python thread runs.
///
/// The calling thread polls every `POLL` to redraw the bar and to check for signals, in both
/// modes. Calls without `verbose` and below `WATCH_FROM` items skip this and run without an extra
/// thread.
///
/// # Errors
///
/// Returns `KeyboardInterrupt` after Ctrl-C, or the error of a failed bar. The workers then stop
/// at the next item and drop what they built, but work that `f` did in place stays done.
pub(crate) fn run_pool_verbose<R: Send, F: FnOnce(&Progress) -> Result<R, Cancelled> + Send>(
    py: Python<'_>,
    pool: &rayon::ThreadPool,
    verbose: bool,
    total: usize,
    desc: &str,
    release_gil: bool,
    f: F,
) -> PyResult<R> {
    let progress = Progress::default();
    if total == 0 || (!verbose && total < WATCH_FROM) {
        let work = || pool.install(|| f(&progress));
        let result = if release_gil { py.detach(work) } else { work() };
        return result.map_err(|Cancelled| interrupted());
    }

    let mut bar = verbose.then(|| Bar::open(py, total, desc)).transpose()?;
    let Watched { result, error } = watch(py, pool, release_gil, &progress, &mut bar, f);
    if let Some(bar) = &bar {
        bar.close();
    }
    match error {
        Some(e) => Err(e),
        None => result.map_err(|Cancelled| interrupted()),
    }
}
