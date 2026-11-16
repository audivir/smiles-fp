use crate::{
    fps::{check_same_bits, extract_fps, get_n_blocks},
    pool::{SyncPtr, run_pool, setup_local_pool},
};
use numpy::{
    PyArray1, PyArray2,
    ndarray::{Array1, Array2},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAny, PyTuple},
};
use rayon::prelude::*;
use std::{cmp::Reverse, collections::BinaryHeap, fs::File, path::PathBuf};

/// Helper struct for top-K sorting
#[derive(Copy, Clone, PartialEq)]
pub(crate) struct SimItem {
    pub(crate) sim: f64,
    pub(crate) idx: u32,
}

impl Eq for SimItem {}

impl Ord for SimItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sim
            .partial_cmp(&other.sim)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for SimItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns 0.0 if no bits are in common and 1.0 if everything matches
#[inline(always)]
pub(crate) fn tanimoto_similarity(pop1: u32, pop2: u32, common: u32) -> f64 {
    let total = pop1 + pop2;
    if total == 0 {
        0.0
    } else if common == total {
        1.0
    } else {
        (common as f64) / ((total - common) as f64)
    }
}

/// Aggregation methods
pub(crate) enum Agg {
    Mean,
    Max,
    Min,
    Full,
    TopK(usize),
}

/// Returns a vector with the number of on bits for each fingerprint.
#[inline(always)]
pub(crate) fn get_popcounts(n_fps: usize, n_blocks: usize, data: &[u64]) -> Vec<u32> {
    if n_blocks == 0 {
        return vec![0; n_fps];
    }
    data.chunks_exact(n_blocks)
        .map(|chunk| chunk.iter().map(|&b| b.count_ones()).sum())
        .collect()
}

/// Parse the aggregation method from user input
pub(crate) fn parse_agg(agg: Option<String>) -> PyResult<Agg> {
    let agg_str = agg.as_deref().unwrap_or("full").to_lowercase();
    let agg = match agg_str.as_str() {
        "mean" => Agg::Mean,
        "max" => Agg::Max,
        "min" => Agg::Min,
        "full" => Agg::Full,
        _ => {
            return Err(PyValueError::new_err(
                "Invalid aggregation type. Choose 'full', 'mean', 'max', or 'min'.",
            ));
        }
    };
    Ok(agg)
}

/// Returns a tuple with the mmap, number of fingerprints, number of bits, and start_index
pub(crate) fn setup_mmap<'py>(
    path: &Bound<'py, PyAny>,
    db_offset: usize,
    db_limit: usize,
) -> PyResult<(memmap2::Mmap, usize, usize, usize)> {
    let file = File::open(path.extract::<PathBuf>()?)?;

    if file.metadata()?.len() < 8 {
        return Err(PyValueError::new_err(
            "Invalid or empty binary footprint file",
        ));
    }

    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let _ = mmap.advise(memmap2::Advice::WillNeed);
    let _ = mmap.advise(memmap2::Advice::Sequential);

    let total_n_fps = u32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;

    if total_n_fps == 0 {
        return Ok((mmap, 0, total_n_fps, 0));
    }

    let n_bits = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;

    // guards against reading past the end of the mapping, as slices built later trust the header.
    let expected_len = get_n_blocks(n_bits)
        .checked_mul(total_n_fps)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(8));
    if expected_len.is_none_or(|expected| mmap.len() < expected) {
        return Err(PyValueError::new_err(format!(
            "Corrupt or truncated fingerprint file: header declares {total_n_fps} fingerprints of \
             {n_bits} bits, but the file has only {} bytes",
            mmap.len()
        )));
    }

    let start_index = db_offset.min(total_n_fps);
    let n_fps = if db_limit == 0 {
        total_n_fps - start_index
    } else {
        db_limit.min(total_n_fps - start_index)
    };

    Ok((mmap, n_fps, n_bits, start_index))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pooled_similarity<'py>(
    py: Python<'py>,
    agg: Agg,
    n_blocks: usize,
    n_fps1: usize,
    n_fps2: usize,
    data1: &[u64],
    data2: &[u64],
    n_threads: i32,
    db_offset: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let popcounts2 = get_popcounts(n_fps2, n_blocks, data2);

    match agg {
        Agg::TopK(k) => {
            let k_actual = k.min(n_fps2);
            let mut out_indices = vec![0u32; n_fps1 * k_actual];
            let mut out_scores = vec![0.0f64; n_fps1 * k_actual];

            if k_actual > 0 {
                run_pool(py, &pool, || {
                    out_indices
                        .par_chunks_mut(k_actual)
                        .zip(out_scores.par_chunks_mut(k_actual))
                        .enumerate()
                        .for_each(|(i, (idx_row, score_row))| {
                            let fp1 = &data1[i * n_blocks..(i + 1) * n_blocks];
                            let pop1: u32 = fp1.iter().map(|b| b.count_ones()).sum();
                            let mut heap = BinaryHeap::with_capacity(k_actual);

                            for j in 0..n_fps2 {
                                let fp2 = &data2[j * n_blocks..(j + 1) * n_blocks];
                                let common = fp1
                                    .iter()
                                    .zip(fp2)
                                    .map(|(b1, b2)| (b1 & b2).count_ones())
                                    .sum::<u32>();
                                let sim = tanimoto_similarity(pop1, popcounts2[j], common);
                                let global_j = (j + db_offset) as u32;

                                if heap.len() < k_actual {
                                    heap.push(Reverse(SimItem { sim, idx: global_j }));
                                } else if sim > heap.peek().unwrap().0.sim {
                                    heap.pop();
                                    heap.push(Reverse(SimItem { sim, idx: global_j }));
                                }
                            }

                            let mut sorted = heap.into_vec();
                            sorted.sort_by_key(|a| Reverse(a.0)); // Descending

                            for rank in 0..k_actual {
                                if rank < sorted.len() {
                                    idx_row[rank] = sorted[rank].0.idx;
                                    score_row[rank] = sorted[rank].0.sim;
                                }
                            }
                        });
                });
            }

            let py_idx = PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n_fps1, k_actual), out_indices).unwrap(),
            );
            let py_scores = PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n_fps1, k_actual), out_scores).unwrap(),
            );

            Ok(PyTuple::new(py, &[py_idx.into_any(), py_scores.into_any()])?.into_any())
        }
        Agg::Mean | Agg::Max | Agg::Min => {
            let mut results = vec![0.0f64; n_fps1];
            run_pool(py, &pool, || {
                results.par_iter_mut().enumerate().for_each(|(i, out)| {
                    let fp1 = &data1[i * n_blocks..(i + 1) * n_blocks];
                    let pop1: u32 = fp1.iter().map(|b| b.count_ones()).sum();

                    let mut local_min = 1.0f64;
                    let mut local_max = 0.0f64;
                    let mut local_sum = 0.0f64;

                    for j in 0..n_fps2 {
                        let fp2 = &data2[j * n_blocks..(j + 1) * n_blocks];
                        let common = fp1
                            .iter()
                            .zip(fp2)
                            .map(|(b1, b2)| (b1 & b2).count_ones())
                            .sum::<u32>();
                        let sim = tanimoto_similarity(pop1, popcounts2[j], common);

                        local_min = local_min.min(sim);
                        local_max = local_max.max(sim);
                        local_sum += sim;
                    }

                    *out = match agg {
                        Agg::Mean => local_sum / (n_fps2 as f64),
                        Agg::Max => local_max,
                        Agg::Min => local_min,
                        _ => 0.0,
                    };
                });
            });
            Ok(PyArray1::from_vec(py, results).into_any())
        }
        Agg::Full => {
            let mut results = vec![0.0f64; n_fps1 * n_fps2];
            run_pool(py, &pool, || {
                results
                    .par_chunks_mut(n_fps2)
                    .enumerate()
                    .for_each(|(i, row)| {
                        let fp1 = &data1[i * n_blocks..(i + 1) * n_blocks];
                        let pop1: u32 = fp1.iter().map(|b| b.count_ones()).sum();

                        for j in 0..n_fps2 {
                            let fp2 = &data2[j * n_blocks..(j + 1) * n_blocks];
                            let common = fp1
                                .iter()
                                .zip(fp2)
                                .map(|(b1, b2)| (b1 & b2).count_ones())
                                .sum::<u32>();
                            row[j] = tanimoto_similarity(pop1, popcounts2[j], common);
                        }
                    });
            });
            Ok(PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((n_fps1, n_fps2), results).unwrap(),
            )
            .into_any())
        }
    }
}

pub(crate) fn pooled_internal_similarity<'py>(
    py: Python<'py>,
    agg: Agg,
    n_blocks: usize,
    n_fps: usize,
    data: &[u64],
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let popcounts = get_popcounts(n_fps, n_blocks, data);

    match agg {
        Agg::TopK(_) => Err(PyValueError::new_err(
            "Top-K is not supported for internal similarity matrices.",
        )),
        Agg::Mean | Agg::Max | Agg::Min => {
            let mut results = vec![0.0f64; n_fps];
            run_pool(py, &pool, || {
                results.par_iter_mut().enumerate().for_each(|(i, out)| {
                    let fp1 = &data[i * n_blocks..(i + 1) * n_blocks];
                    let pop1 = popcounts[i];

                    let mut local_min = 1.0f64;
                    let mut local_max = 0.0f64;
                    let mut local_sum = 0.0f64;

                    for j in 0..n_fps {
                        if i == j {
                            // skip self
                            continue;
                        }

                        let fp2 = &data[j * n_blocks..(j + 1) * n_blocks];
                        let common = fp1
                            .iter()
                            .zip(fp2)
                            .map(|(b1, b2)| (b1 & b2).count_ones())
                            .sum::<u32>();
                        let sim = tanimoto_similarity(pop1, popcounts[j], common);

                        local_min = local_min.min(sim);
                        local_max = local_max.max(sim);
                        local_sum += sim;
                    }

                    *out = if n_fps < 2 {
                        0.0
                    } else {
                        match agg {
                            Agg::Mean => local_sum / (n_fps - 1) as f64,
                            Agg::Max => local_max,
                            Agg::Min => local_min,
                            _ => 0.0,
                        }
                    };
                });
            });
            Ok(PyArray1::from_vec(py, results).into_any())
        }
        Agg::Full => {
            let n_pairs = n_fps * (n_fps - 1) / 2;
            let mut results = vec![0.0f64; n_pairs];
            let res_ptr = SyncPtr(results.as_mut_ptr());

            run_pool(py, &pool, || {
                (0..n_fps).into_par_iter().for_each(move |i| {
                    let ptr = res_ptr; // forces the closure to capture the thread-safe SyncPtr

                    let offset = i * n_fps - i * (i + 1) / 2;
                    let fp1 = &data[i * n_blocks..(i + 1) * n_blocks];
                    let pop1 = popcounts[i];

                    for j in (i + 1)..n_fps {
                        let fp2 = &data[j * n_blocks..(j + 1) * n_blocks];
                        let common = fp1
                            .iter()
                            .zip(fp2)
                            .map(|(b1, b2)| (b1 & b2).count_ones())
                            .sum::<u32>();
                        let sim = tanimoto_similarity(pop1, popcounts[j], common);

                        unsafe {
                            std::ptr::write(ptr.0.add(offset + (j - i - 1)), sim);
                        }
                    }
                });
            });
            Ok(PyArray1::from_vec(py, results).into_any())
        }
    }
}

#[pyfunction]
#[pyo3(signature = (py_fps, py_fps2, n_threads=-1, agg=None))]
pub(crate) fn bulk_tanimoto_parallel<'py>(
    py: Python<'py>,
    py_fps: &Bound<'py, PyAny>,
    py_fps2: &Bound<'py, PyAny>,
    n_threads: i32,
    agg: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let agg = parse_agg(agg)?;
    let fps1 = extract_fps(py, py_fps, n_threads)?;
    let fps2 = extract_fps(py, py_fps2, n_threads)?;

    let n_fps1 = fps1.n_fps;
    let n_fps2 = fps2.n_fps;

    if n_fps1 == 0 || n_fps2 == 0 {
        return match agg {
            Agg::Full => Ok(
                PyArray2::from_owned_array(py, Array2::<f64>::zeros((n_fps1, n_fps2))).into_any(),
            ),
            _ => Ok(PyArray1::from_owned_array(py, Array1::<f64>::zeros(n_fps1)).into_any()),
        };
    }

    check_same_bits(&fps1, &fps2)?;

    pooled_similarity(
        py,
        agg,
        fps1.n_blocks,
        n_fps1,
        n_fps2,
        fps1.blocks(),
        fps2.blocks(),
        n_threads,
        0,
    )
}

#[pyfunction]
#[pyo3(signature = (py_fps, py_fps2, k=10, n_threads=-1))]
pub(crate) fn bulk_tanimoto_parallel_topk<'py>(
    py: Python<'py>,
    py_fps: &Bound<'py, PyAny>,
    py_fps2: &Bound<'py, PyAny>,
    k: usize,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let fps1 = extract_fps(py, py_fps, n_threads)?;
    let fps2 = extract_fps(py, py_fps2, n_threads)?;

    let n_fps1 = fps1.n_fps;
    let n_fps2 = fps2.n_fps;

    if n_fps1 == 0 || n_fps2 == 0 {
        let py_idx = PyArray2::from_owned_array(py, Array2::<u32>::zeros((n_fps1, 0)));
        let py_scores = PyArray2::from_owned_array(py, Array2::<f64>::zeros((n_fps1, 0)));
        return Ok(PyTuple::new(py, &[py_idx.into_any(), py_scores.into_any()])?.into_any());
    }

    check_same_bits(&fps1, &fps2)?;

    pooled_similarity(
        py,
        Agg::TopK(k),
        fps1.n_blocks,
        n_fps1,
        n_fps2,
        fps1.blocks(),
        fps2.blocks(),
        n_threads,
        0,
    )
}

#[pyfunction]
#[pyo3(signature = (path1, path2, n_threads=-1, agg=None, db_offset=0, db_limit=0))]
pub(crate) fn bulk_tanimoto_mmap<'py>(
    py: Python<'py>,
    path1: &Bound<'py, PyAny>,
    path2: &Bound<'py, PyAny>,
    n_threads: i32,
    agg: Option<String>,
    db_offset: usize,
    db_limit: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let agg = parse_agg(agg)?;
    let (mmap1, n_fps1, n_bits1, _) = setup_mmap(path1, 0, 0)?;
    let (mmap2, n_fps2, n_bits2, start_index) = setup_mmap(path2, db_offset, db_limit)?;

    if n_fps1 == 0 || n_fps2 == 0 {
        return match agg {
            Agg::Full => Ok(
                PyArray2::from_owned_array(py, Array2::<f64>::zeros((n_fps1, n_fps2))).into_any(),
            ),
            _ => Ok(PyArray1::from_owned_array(py, Array1::<f64>::zeros(n_fps1)).into_any()),
        };
    }

    if n_bits1 != n_bits2 {
        return Err(PyValueError::new_err("Fingerprint bit mismatch"));
    }

    let n_blocks = get_n_blocks(n_bits1);

    let data1 = unsafe {
        std::slice::from_raw_parts(mmap1.as_ptr().add(8) as *const u64, n_fps1 * n_blocks)
    };
    let byte_offset = 8 + (start_index * n_blocks * 8);
    let data2 = unsafe {
        std::slice::from_raw_parts(
            mmap2.as_ptr().add(byte_offset) as *const u64,
            n_fps2 * n_blocks,
        )
    };

    pooled_similarity(
        py,
        agg,
        n_blocks,
        n_fps1,
        n_fps2,
        data1,
        data2,
        n_threads,
        start_index,
    )
}

#[pyfunction]
#[pyo3(signature = (path1, path2, k=10, n_threads=-1, db_offset=0, db_limit=0))]
pub(crate) fn bulk_tanimoto_mmap_topk<'py>(
    py: Python<'py>,
    path1: &Bound<'py, PyAny>,
    path2: &Bound<'py, PyAny>,
    k: usize,
    n_threads: i32,
    db_offset: usize,
    db_limit: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let (mmap1, n_fps1, n_bits1, _) = setup_mmap(path1, 0, 0)?;
    let (mmap2, n_fps2, n_bits2, start_index) = setup_mmap(path2, db_offset, db_limit)?;

    if n_fps1 == 0 || n_fps2 == 0 {
        let py_idx = PyArray2::from_owned_array(py, Array2::<u32>::zeros((n_fps1, 0)));
        let py_scores = PyArray2::from_owned_array(py, Array2::<f64>::zeros((n_fps1, 0)));
        return Ok(PyTuple::new(py, &[py_idx.into_any(), py_scores.into_any()])?.into_any());
    }

    if n_bits1 != n_bits2 {
        return Err(PyValueError::new_err("Fingerprint bit mismatch"));
    }

    let n_blocks = get_n_blocks(n_bits1);

    let data1 = unsafe {
        std::slice::from_raw_parts(mmap1.as_ptr().add(8) as *const u64, n_fps1 * n_blocks)
    };
    let byte_offset = 8 + (start_index * n_blocks * 8);
    let data2 = unsafe {
        std::slice::from_raw_parts(
            mmap2.as_ptr().add(byte_offset) as *const u64,
            n_fps2 * n_blocks,
        )
    };

    pooled_similarity(
        py,
        Agg::TopK(k),
        n_blocks,
        n_fps1,
        n_fps2,
        data1,
        data2,
        n_threads,
        start_index,
    )
}

#[pyfunction]
#[pyo3(signature = (py_fps, n_threads=-1, agg=None))]
pub(crate) fn internal_tanimoto_parallel<'py>(
    py: Python<'py>,
    py_fps: &Bound<'py, PyAny>,
    n_threads: i32,
    agg: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let agg = parse_agg(agg)?;
    let fps = extract_fps(py, py_fps, n_threads)?;
    let n_fps = fps.n_fps;

    if n_fps == 0 {
        return match agg {
            Agg::Full => Ok(PyArray1::from_owned_array(py, Array1::<f64>::zeros(0)).into_any()),
            _ => Ok(PyArray1::from_owned_array(py, Array1::<f64>::zeros(0)).into_any()),
        };
    }

    pooled_internal_similarity(py, agg, fps.n_blocks, n_fps, fps.blocks(), n_threads)
}

#[pyfunction]
#[pyo3(signature = (path, n_threads=-1, agg=None))]
pub(crate) fn internal_tanimoto_mmap<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    n_threads: i32,
    agg: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let agg = parse_agg(agg)?;
    let (mmap, n_fps, n_bits, _) = setup_mmap(path, 0, 0)?;

    if n_fps == 0 {
        return match agg {
            Agg::Full => Ok(PyArray1::from_owned_array(py, Array1::<f64>::zeros(0)).into_any()),
            _ => Ok(PyArray1::from_owned_array(py, Array1::<f64>::zeros(0)).into_any()),
        };
    }

    let n_blocks = get_n_blocks(n_bits);

    let data =
        unsafe { std::slice::from_raw_parts(mmap.as_ptr().add(8) as *const u64, n_fps * n_blocks) };
    pooled_internal_similarity(py, agg, n_blocks, n_fps, data, n_threads)
}
