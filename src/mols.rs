use crate::{
    ffi, ffi_error,
    pool::{run_pool, setup_local_pool},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PySequence},
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Target number of write units per thread before `mols_to_smiles` stops splitting molecules.
pub(crate) const UNITS_PER_THREAD: usize = 4;

/// Result of one write: task index, slot within the task, and the SMILES or error message.
pub(crate) type WrittenItem = (usize, usize, Result<String, String>);

/// Container for one `ROMol` (original or private copy) and the (task, slot, root) items it writes.
pub(crate) struct WriteUnit {
    pub(crate) ptr: usize,
    pub(crate) items: Vec<(usize, usize, i32)>,
}

/// Wrapper around private molecule copies that frees them on drop.
pub(crate) struct OwnedMols(pub(crate) Vec<usize>);

impl Drop for OwnedMols {
    fn drop(&mut self) {
        for &p in &self.0 {
            ffi::free_mol_raw(p);
        }
    }
}

/// Expands `rooted_at_atom` to one root atom per molecule, with -1 meaning no rooting.
///
/// # Errors
///
/// Returns `Err` if the argument is not an int, `None`, or a sequence of `int | None`, or if the
/// sequence length differs from the number of molecules.
pub(crate) fn parse_roots(arg: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Vec<i32>> {
    let Some(arg) = arg.filter(|a| !a.is_none()) else {
        return Ok(vec![-1; n]);
    };
    if let Ok(root) = arg.extract::<i32>() {
        return Ok(vec![root; n]);
    }
    let roots: Vec<Option<i32>> = arg.extract().map_err(|_| {
        PyTypeError::new_err("rooted_at_atom must be an int, None or a sequence of int | None")
    })?;
    if roots.len() != n {
        return Err(PyValueError::new_err(format!(
            "rooted_at_atom has {} items for {n} molecules",
            roots.len()
        )));
    }
    Ok(roots.into_iter().map(|r| r.unwrap_or(-1)).collect())
}

/// Writes SMILES for every molecule on real threads with GIL held.
///
/// The `doRandom` option is unsupported because the random generator of RDKit is global and not
/// thread-safe.
#[pyfunction]
#[pyo3(signature = (
    mols,
    isomeric_smiles=true,
    kekule_smiles=false,
    rooted_at_atom=None,
    canonical=true,
    all_bonds_explicit=false,
    all_hs_explicit=false,
    *,
    ignore_atom_map_numbers=false,
    n_threads=-1
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mols_to_smiles<'py>(
    py: Python<'py>,
    mols: &Bound<'py, PySequence>,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    rooted_at_atom: Option<&Bound<'py, PyAny>>,
    canonical: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyList>> {
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let batch =
        ffi::extract_mol_ptrs(mols.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))?;
    // owns the reference to the molecule sequence until all SMILES are written.
    let keepalive: Bound<'py, PyAny> =
        unsafe { Bound::from_owned_ptr(py, batch.keepalive as *mut pyo3::ffi::PyObject) };
    let roots = parse_roots(rooted_at_atom, batch.ptrs.len())?;

    // `MolToSmiles` aborts or segfaults when threads share one `ROMol`, so each distinct molecule
    // is one task for one thread. That thread writes every distinct root requested for the
    // molecule, and results are fanned out to all positions.
    let mut task_of: HashMap<usize, usize> = HashMap::with_capacity(batch.ptrs.len());
    let mut cell_of: HashMap<(usize, i32), usize> = HashMap::new();
    let mut tasks: Vec<(usize, Vec<i32>)> = Vec::new();
    let positions: Vec<(usize, usize)> = batch
        .ptrs
        .iter()
        .zip(&roots)
        .map(|(&ptr, &root)| {
            let t = *task_of.entry(ptr).or_insert_with(|| {
                tasks.push((ptr, Vec::new()));
                tasks.len() - 1
            });
            let k = *cell_of.entry((t, root)).or_insert_with(|| {
                tasks[t].1.push(root);
                tasks[t].1.len() - 1
            });
            (t, k)
        })
        .collect();
    // private copies let several threads write one molecule, one chunk of roots per copy. Copies
    // are made here, serially, as copying a shared `ROMol` from several threads is not known to be
    // safe. They are made only when distinct molecules cannot keep all threads busy, because
    // otherwise they add serial work.
    let n_pool_threads = pool.current_num_threads();
    let copies_wanted = (UNITS_PER_THREAD * n_pool_threads).div_ceil(tasks.len().max(1));
    let mut owned = OwnedMols(Vec::new());
    let mut units: Vec<WriteUnit> = Vec::with_capacity(tasks.len());
    for (t, (ptr, task_roots)) in tasks.iter().enumerate() {
        let n_chunks = copies_wanted.min(task_roots.len()).max(1);
        let chunk_len = task_roots.len().div_ceil(n_chunks);
        for (c, chunk) in task_roots.chunks(chunk_len).enumerate() {
            let unit_ptr = if c == 0 {
                *ptr
            } else {
                let copy = ffi::clone_mol_raw(*ptr)
                    .map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;
                owned.0.push(copy);
                copy
            };
            units.push(WriteUnit {
                ptr: unit_ptr,
                items: chunk
                    .iter()
                    .enumerate()
                    .map(|(j, &root)| (t, c * chunk_len + j, root))
                    .collect(),
            });
        }
    }
    // the GIL stays held, so no other Python thread can modify the molecules during the write.
    let written: Vec<Vec<WrittenItem>> = pool.install(|| {
        // each unit owns its `ROMol`, so no two threads touch the same one.
        units
            .par_iter()
            .map(|unit| {
                unit.items
                    .iter()
                    .map(|&(t, k, root)| {
                        let opts = ffi::SmilesOpts {
                            isomeric_smiles,
                            kekule_smiles,
                            rooted_at_atom: root,
                            canonical,
                            all_bonds_explicit,
                            all_hs_explicit,
                            ignore_atom_map_numbers,
                        };
                        let smi = ffi::mol_to_smiles_raw(unit.ptr, &opts)
                            .map_err(|e| e.what().to_string());
                        (t, k, smi)
                    })
                    .collect()
            })
            .collect()
    });
    let mut results: Vec<Vec<Option<Result<String, String>>>> =
        tasks.iter().map(|(_, r)| vec![None; r.len()]).collect();
    for (t, k, smi) in written.into_iter().flatten() {
        results[t][k] = Some(smi);
    }
    drop(keepalive);

    let mut smiles = Vec::with_capacity(positions.len());
    for (i, &(t, k)) in positions.iter().enumerate() {
        match results[t][k]
            .as_ref()
            .expect("every (molecule, root) is written")
        {
            Ok(smi) => smiles.push(smi.clone()),
            Err(e) => {
                return Err(PyRuntimeError::new_err(format!(
                    "MolToSmiles failed for molecule {i}: {e}"
                )));
            }
        }
    }
    PyList::new(py, smiles)
}

/// Parses SMILES strings on real threads with GIL released.
///
/// Returns `None` for each SMILES that fails to parse.
#[pyfunction]
#[pyo3(signature = (smiles, sanitize=true, *, n_threads=-1))]
pub(crate) fn mols_from_smiles<'py>(
    py: Python<'py>,
    smiles: Vec<String>,
    sanitize: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let ptrs: Vec<usize> = run_pool(py, &pool, || {
        smiles
            .par_iter()
            .map(|s| ffi::parse_smiles_raw(s, sanitize))
            .collect()
    });
    let ptr_val = ffi::wrap_mols(&ptrs);
    if ptr_val == 0 {
        return Err(PyErr::take(py)
            .unwrap_or_else(|| PyRuntimeError::new_err("Failed to wrap molecules in C++")));
    }
    let list = unsafe { Bound::from_owned_ptr(py, ptr_val as *mut pyo3::ffi::PyObject) };
    Ok(list.into_any())
}
