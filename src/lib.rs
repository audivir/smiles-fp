use crate::ffi::FpBatch;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
    ndarray::{Array1, Array2},
};
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PyModule, PySequence, PyTuple},
};
use rayon::prelude::*;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fs::File,
    io::{BufWriter, Read, Write},
    path::PathBuf,
};

mod clean;
mod salt_remover;

/// Thread-safe pointer wrapper
#[derive(Copy, Clone)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

/// Helper struct for top-K sorting
#[derive(Copy, Clone, PartialEq)]
struct SimItem {
    sim: f64,
    idx: u32,
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
fn tanimoto_similarity(pop1: u32, pop2: u32, common: u32) -> f64 {
    let total = pop1 + pop2;
    if total == 0 {
        0.0
    } else if common == total {
        1.0
    } else {
        (common as f64) / ((total - common) as f64)
    }
}

/// Get the actual number of threads
#[inline(always)]
fn get_n_threads(n_threads: i32) -> usize {
    if n_threads > 0 {
        n_threads as usize
    } else {
        rayon::current_num_threads()
    }
}

/// Set up a local Rayon thread pool
#[inline(always)]
fn setup_local_pool(n_threads: i32) -> Result<rayon::ThreadPool, String> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(get_n_threads(n_threads))
        .build()
        .map_err(|e| format!("Failed to build thread pool: {}", e))
}

/// Runs `f` on Rayon pool with GIL released, so `f` must not touch Python.
fn run_pool<R: Send, F: FnOnce() -> R + Send>(py: Python<'_>, pool: &rayon::ThreadPool, f: F) -> R {
    py.detach(|| pool.install(f))
}

/// Aggregation methods
enum Agg {
    Mean,
    Max,
    Min,
    Full,
    TopK(usize),
}

/// FFI boundary with C++
#[cxx::bridge]
mod ffi {
    /// Container for `Chem.MolToSmiles` arguments, in the same order as RDKit.
    struct SmilesOpts {
        isomeric_smiles: bool,
        kekule_smiles: bool,
        rooted_at_atom: i32,
        canonical: bool,
        all_bonds_explicit: bool,
        all_hs_explicit: bool,
        ignore_atom_map_numbers: bool,
    }

    /// Container for the flags of the cleaning pipeline.
    #[derive(Clone, Copy)]
    struct CleanOpts {
        no_salts: bool,
        no_isotopes: bool,
        no_stereo: bool,
        isomeric: bool,
        neutralize: bool,
    }

    /// Container for a cleaned SMILES. An empty `smiles` marks an invalid input.
    struct Cleaned {
        smiles: String,
        mol: usize,
    }

    struct MolPtrs {
        keepalive: usize,
        ptrs: Vec<usize>,
    }

    struct FpBatch {
        n_fps: u32,
        n_bits: u32,
        data: Vec<u64>,
    }

    unsafe extern "C++" {
        include!("smiles_fp/src/rdkit_shim.h");

        fn write_rdkit_bit_vects_to_file(py_list_ptr: usize, filename: &str) -> Result<()>;
        fn extract_rdkit_bit_vects(py_list_ptr: usize) -> Result<FpBatch>;
        fn create_rdkit_bit_vects(n_fps: u32, n_bits: u32, blocks: &[u64]) -> usize;
        fn parse_smiles_raw(smiles: &str, sanitize: bool) -> usize;
        fn wrap_mols(ptrs: &[usize]) -> usize;
        fn extract_mol_ptrs(py_seq_ptr: usize) -> Result<MolPtrs>;
        fn mol_to_smiles_raw(mol_ptr: usize, opts: &SmilesOpts) -> Result<String>;
        fn clone_mol_raw(mol_ptr: usize) -> Result<usize>;
        fn free_mol_raw(mol_ptr: usize);
        fn make_morgan_generator(radius: u32, n_bits: u32) -> Result<usize>;
        fn free_morgan_generator(generator: usize);
        fn make_salt_stripper(smarts_lines: &str) -> Result<usize>;
        fn free_salt_stripper(stripper: usize);
        fn make_cleaner(opts: &CleanOpts, allowed: &[u32], salt_lines: &str) -> Result<usize>;
        fn free_cleaner(cleaner: usize);
        fn clone_rwmol_raw(mol_ptr: usize) -> Result<usize>;
        fn clean_mol_raw(cleaner: usize, mol_ptr: usize, want_smiles: bool) -> Cleaned;
        fn clean_smiles_raw(cleaner: usize, smiles: &str, want_mol: bool) -> Cleaned;
        fn strip_salts_raw(stripper: usize, mol_ptr: usize, keep_last: bool) -> usize;
        fn smiles_to_fp_blocks(generator: usize, smiles: &str, out: &mut [u64]) -> i32;
    }
}

/// Converts C++ exception into Python error.
///
/// Returns the pending Python exception if C++ left one, which keeps its original type and message.
fn ffi_error(py: Python<'_>, e: cxx::Exception, os_error: bool) -> PyErr {
    if let Some(err) = PyErr::take(py) {
        return err;
    }
    let msg = e.what().to_string();
    if os_error {
        PyOSError::new_err(msg)
    } else {
        PyRuntimeError::new_err(msg)
    }
}

impl FpBatch {
    /// Extracts fingerprints from Python sequence, raising on invalid input.
    #[inline(always)]
    fn from_py_seq(py_seq: &Bound<'_, PySequence>) -> PyResult<Self> {
        let py = py_seq.py();
        ffi::extract_rdkit_bit_vects(py_seq.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))
    }
}

/// Returns the number of u64 blocks needed (ceil)
#[inline(always)]
fn get_n_blocks(n_bits: usize) -> usize {
    n_bits.div_ceil(64)
}

/// Fingerprint blocks that are either owned or borrowed from a NumPy array.
enum FpData<'py> {
    Owned(Vec<u64>),
    Borrowed(PyReadonlyArray2<'py, u64>),
}

/// Fingerprints of any accepted input as rows of 64-bit blocks.
struct Fps<'py> {
    n_fps: usize,
    /// Bits per fingerprint, unknown for packed arrays whose last block may be padded.
    n_bits: Option<usize>,
    n_blocks: usize,
    data: FpData<'py>,
}

impl Fps<'_> {
    fn blocks(&self) -> &[u64] {
        match &self.data {
            FpData::Owned(v) => v,
            FpData::Borrowed(a) => a.as_slice().expect("checked to be contiguous"),
        }
    }
}

/// Packs 64 bytes of 0 or 1 into one block, element `k` becoming bit `k`.
///
/// Returns the block and the OR of all input words, which has bits above 0 set if a byte is
/// not 0 or 1.
#[inline(always)]
fn pack_block(bytes: &[u8; 64]) -> (u64, u64) {
    let mut block = 0u64;
    let mut any = 0u64;
    for m in 0..8 {
        let x = u64::from_le_bytes(bytes[m * 8..m * 8 + 8].try_into().unwrap());
        any |= x;
        // each byte of `x` is 0 or 1, so the multiplication gathers them into the top byte.
        block |= (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * m);
    }
    (block, any)
}

/// Packs one fingerprint of 0/1 bytes into blocks and returns whether all values were 0 or 1.
fn pack_row(row: &[u8], out: &mut [u64]) -> bool {
    const NOT_BIT: u64 = 0xFEFE_FEFE_FEFE_FEFE;
    let mut any = 0u64;
    let (chunks, rest) = row.as_chunks::<64>();
    for (block, chunk) in out.iter_mut().zip(chunks) {
        let (b, a) = pack_block(chunk);
        *block = b;
        any |= a;
    }
    if !rest.is_empty() {
        let mut buf = [0u8; 64];
        buf[..rest.len()].copy_from_slice(rest);
        let (b, a) = pack_block(&buf);
        out[row.len() / 64] = b;
        any |= a;
    }
    any & NOT_BIT == 0
}

/// Extracts fingerprints from RDKit bit vectors, a 2D uint8 array of bits, or a 2D uint64 array
/// of packed blocks.
fn extract_fps<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    n_threads: i32,
) -> PyResult<Fps<'py>> {
    if let Ok(arr) = obj.cast::<PyArray2<u8>>() {
        let arr = arr.readonly();
        let (n_fps, n_bits) = (arr.shape()[0], arr.shape()[1]);
        let n_blocks = get_n_blocks(n_bits);
        if n_fps == 0 {
            return Ok(Fps {
                n_fps,
                n_bits: Some(n_bits),
                n_blocks,
                data: FpData::Owned(vec![]),
            });
        }
        if n_bits == 0 {
            return Err(PyValueError::new_err(
                "Fingerprints must have at least one bit",
            ));
        }
        let mut data = vec![0u64; n_fps * n_blocks];
        let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
        // `as_slice` also succeeds for column-major arrays, whose rows are not contiguous.
        let valid = match arr.as_slice() {
            Ok(src) if arr.is_c_contiguous() => run_pool(py, &pool, || {
                data.par_chunks_mut(n_blocks)
                    .zip(src.par_chunks(n_bits))
                    .map(|(out, row)| pack_row(row, out))
                    .all(|ok| ok)
            }),
            _ => {
                // strided rows are gathered one at a time, so no full copy of the array is made.
                let view = arr.as_array();
                run_pool(py, &pool, || {
                    data.par_chunks_mut(n_blocks)
                        .enumerate()
                        .map_init(
                            || vec![0u8; n_bits],
                            |buf, (i, out)| {
                                for (dst, &v) in buf.iter_mut().zip(view.row(i)) {
                                    *dst = v;
                                }
                                pack_row(buf, out)
                            },
                        )
                        .all(|ok| ok)
                })
            }
        };
        if !valid {
            return Err(PyValueError::new_err(
                "Fingerprint arrays must contain only 0 and 1",
            ));
        }
        return Ok(Fps {
            n_fps,
            n_bits: Some(n_bits),
            n_blocks,
            data: FpData::Owned(data),
        });
    }
    if let Ok(arr) = obj.cast::<PyArray2<u64>>() {
        let arr = arr.readonly();
        let (n_fps, n_blocks) = (arr.shape()[0], arr.shape()[1]);
        let data = if arr.is_c_contiguous() {
            FpData::Borrowed(arr)
        } else {
            FpData::Owned(arr.as_array().iter().copied().collect())
        };
        return Ok(Fps {
            n_fps,
            n_bits: None,
            n_blocks,
            data,
        });
    }
    if obj.cast::<PyUntypedArray>().is_ok() {
        return Err(PyTypeError::new_err(
            "Fingerprint arrays must be 2D with dtype uint8 (bits) or uint64 (packed blocks)",
        ));
    }
    let batch = FpBatch::from_py_seq(obj.cast::<PySequence>()?)?;
    Ok(Fps {
        n_fps: batch.n_fps as usize,
        n_bits: Some(batch.n_bits as usize),
        n_blocks: get_n_blocks(batch.n_bits as usize),
        data: FpData::Owned(batch.data),
    })
}

/// Raises if two fingerprint sets cannot be compared.
fn check_same_bits(a: &Fps<'_>, b: &Fps<'_>) -> PyResult<()> {
    let bits_differ = matches!((a.n_bits, b.n_bits), (Some(x), Some(y)) if x != y);
    if bits_differ || a.n_blocks != b.n_blocks {
        return Err(PyValueError::new_err("Fingerprint bit mismatch"));
    }
    Ok(())
}

/// Returns a vector with the number of on bits for each fingerprint.
#[inline(always)]
fn get_popcounts(n_fps: usize, n_blocks: usize, data: &[u64]) -> Vec<u32> {
    if n_blocks == 0 {
        return vec![0; n_fps];
    }
    data.chunks_exact(n_blocks)
        .map(|chunk| chunk.iter().map(|&b| b.count_ones()).sum())
        .collect()
}

/// Parse the aggregation method from user input
fn parse_agg(agg: Option<String>) -> PyResult<Agg> {
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
fn setup_mmap<'py>(
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
fn pooled_similarity<'py>(
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

fn pooled_internal_similarity<'py>(
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
#[pyo3(name = "_save_fingerprints")]
fn save_fingerprints<'py>(fps: &Bound<'py, PySequence>, path: &Bound<'py, PyAny>) -> PyResult<()> {
    let py = fps.py();
    let path_buf: PathBuf = path.extract()?;
    let path_str = path_buf
        .to_str()
        .ok_or_else(|| PyValueError::new_err("path is not valid UTF-8"))?;
    ffi::write_rdkit_bit_vects_to_file(fps.as_ptr() as usize, path_str)
        .map_err(|e| ffi_error(py, e, true))
}

/// Copies the bits of RDKit fingerprints into a 2D array of little-endian 64-bit blocks.
#[pyfunction]
#[pyo3(name = "_fingerprints_to_blocks")]
fn fingerprints_to_blocks<'py>(
    py: Python<'py>,
    py_fps: &Bound<'py, PySequence>,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let batch = FpBatch::from_py_seq(py_fps)?;
    let shape = (batch.n_fps as usize, get_n_blocks(batch.n_bits as usize));
    let blocks = Array2::from_shape_vec(shape, batch.data)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, blocks))
}

#[pyfunction]
#[pyo3(name = "_load_fingerprints")]
fn load_fingerprints<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let path: PathBuf = path.extract()?;
    let mut file = File::open(path)?;

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let n_fps = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let n_bits = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let n_blocks = n_bits.div_ceil(64);

    if n_fps == 0 {
        return Ok(PyList::empty(py).into_any());
    }

    let n_words = (n_fps as usize) * (n_blocks as usize);
    let expected_len = n_words.checked_mul(8).and_then(|n| n.checked_add(8));
    if expected_len
        .is_none_or(|expected| file.metadata().map_or(true, |m| m.len() < expected as u64))
    {
        return Err(PyValueError::new_err(format!(
            "Corrupt or truncated fingerprint file: header declares {n_fps} fingerprints of \
             {n_bits} bits"
        )));
    }

    let mut data = vec![0u64; n_words];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 8) };
    file.read_exact(bytes)?;

    let ptr_val = ffi::create_rdkit_bit_vects(n_fps, n_bits, &data);
    if ptr_val == 0 {
        return Err(PyErr::take(py).unwrap_or_else(|| {
            PyRuntimeError::new_err("Failed to construct fingerprints in C++")
        }));
    }

    let py_list = unsafe { Bound::from_owned_ptr(py, ptr_val as *mut pyo3::ffi::PyObject) };
    Ok(py_list.into_any())
}

/// Computes Morgan fingerprints of SMILES strings on real threads and saves them to a file.
///
/// No Python objects are created, so memory stays at the size of the fingerprints.
///
/// # Errors
///
/// Returns `Err` if a SMILES cannot be parsed, naming the first one, or if the file cannot be
/// written. No file is left behind on failure.
#[pyfunction]
#[pyo3(signature = (smiles, path, radius=2, n_bits=2048, *, n_threads=-1))]
fn save_morgan_fps_from_smiles(
    py: Python<'_>,
    smiles: Vec<String>,
    path: PathBuf,
    radius: u32,
    n_bits: u32,
    n_threads: i32,
) -> PyResult<()> {
    if n_bits == 0 {
        return Err(PyValueError::new_err("n_bits must be positive"));
    }
    let n_fps = u32::try_from(smiles.len()).map_err(|_| {
        PyValueError::new_err("Too many fingerprints for the file format (max 2^32-1)")
    })?;
    let n_blocks = (n_bits as usize).div_ceil(64);
    let pool = setup_local_pool(n_threads).map_err(PyRuntimeError::new_err)?;
    let generator = ffi::make_morgan_generator(radius, n_bits)
        .map_err(|e| PyRuntimeError::new_err(e.what().to_string()))?;

    let mut blocks = vec![0u64; smiles.len() * n_blocks];
    let failed: Vec<bool> = run_pool(py, &pool, || {
        blocks
            .par_chunks_mut(n_blocks)
            .zip(smiles.par_iter())
            .map(|(out, smi)| ffi::smiles_to_fp_blocks(generator, smi, out) != 0)
            .collect()
    });
    ffi::free_morgan_generator(generator);
    if let Some(i) = failed.iter().position(|&f| f) {
        return Err(PyValueError::new_err(format!(
            "Could not convert SMILES {i}: {}",
            smiles[i]
        )));
    }

    let write = || -> std::io::Result<()> {
        let mut out = BufWriter::new(File::create(&path)?);
        out.write_all(&n_fps.to_le_bytes())?;
        out.write_all(&(if n_fps == 0 { 0 } else { n_bits }).to_le_bytes())?;
        for block in &blocks {
            out.write_all(&block.to_le_bytes())?;
        }
        out.flush()
    };
    write().inspect_err(|_| {
        let _ = std::fs::remove_file(&path);
    })?;
    Ok(())
}

/// Target number of write units per thread before `mols_to_smiles` stops splitting molecules.
const UNITS_PER_THREAD: usize = 4;

/// Result of one write: task index, slot within the task, and the SMILES or error message.
type WrittenItem = (usize, usize, Result<String, String>);

/// Container for one `ROMol` (original or private copy) and the (task, slot, root) items it writes.
struct WriteUnit {
    ptr: usize,
    items: Vec<(usize, usize, i32)>,
}

/// Wrapper around private molecule copies that frees them on drop.
struct OwnedMols(Vec<usize>);

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
fn parse_roots(arg: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Vec<i32>> {
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
fn mols_to_smiles<'py>(
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
fn mols_from_smiles<'py>(
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

#[pyfunction]
#[pyo3(signature = (py_fps, py_fps2, n_threads=-1, agg=None))]
fn bulk_tanimoto_parallel<'py>(
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
fn bulk_tanimoto_parallel_topk<'py>(
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
fn bulk_tanimoto_mmap<'py>(
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
fn bulk_tanimoto_mmap_topk<'py>(
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
fn internal_tanimoto_parallel<'py>(
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
fn internal_tanimoto_mmap<'py>(
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

#[pymodule]
fn _smiles_fp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_fingerprints, m)?)?;
    m.add_function(wrap_pyfunction!(load_fingerprints, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprints_to_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(mols_from_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(mols_to_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(salt_remover::strip_salts, m)?)?;
    m.add_function(wrap_pyfunction!(clean::clean, m)?)?;
    m.add_function(wrap_pyfunction!(save_morgan_fps_from_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_tanimoto_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_tanimoto_parallel_topk, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_tanimoto_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(bulk_tanimoto_mmap_topk, m)?)?;
    m.add_function(wrap_pyfunction!(internal_tanimoto_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(internal_tanimoto_mmap, m)?)?;
    Ok(())
}
