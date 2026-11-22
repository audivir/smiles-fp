use crate::{
    ffi,
    ffi::FpBatch,
    ffi_error,
    pool::{Cancelled, run_pool, run_pool_verbose, setup_local_pool},
};
use numpy::{
    PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
    ndarray::Array2,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAny, PyList, PySequence},
};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::PathBuf,
};

impl FpBatch {
    /// Extracts fingerprints from Python sequence, raising on invalid input.
    #[inline(always)]
    pub(crate) fn from_py_seq(py_seq: &Bound<'_, PySequence>) -> PyResult<Self> {
        let py = py_seq.py();
        ffi::extract_rdkit_bit_vects(py_seq.as_ptr() as usize).map_err(|e| ffi_error(py, e, false))
    }
}

/// Returns the number of u64 blocks needed (ceil)
#[inline(always)]
pub(crate) fn get_n_blocks(n_bits: usize) -> usize {
    n_bits.div_ceil(64)
}

/// Fingerprint blocks that are either owned or borrowed from a NumPy array.
pub(crate) enum FpData<'py> {
    Owned(Vec<u64>),
    Borrowed(PyReadonlyArray2<'py, u64>),
}

/// Fingerprints of any accepted input as rows of 64-bit blocks.
pub(crate) struct Fps<'py> {
    pub(crate) n_fps: usize,
    /// Bits per fingerprint, unknown for packed arrays whose last block may be padded.
    pub(crate) n_bits: Option<usize>,
    pub(crate) n_blocks: usize,
    pub(crate) data: FpData<'py>,
}

impl Fps<'_> {
    pub(crate) fn blocks(&self) -> &[u64] {
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
pub(crate) fn pack_block(bytes: &[u8; 64]) -> (u64, u64) {
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
pub(crate) fn pack_row(row: &[u8], out: &mut [u64]) -> bool {
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
pub(crate) fn extract_fps<'py>(
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
pub(crate) fn check_same_bits(a: &Fps<'_>, b: &Fps<'_>) -> PyResult<()> {
    let bits_differ = matches!((a.n_bits, b.n_bits), (Some(x), Some(y)) if x != y);
    if bits_differ || a.n_blocks != b.n_blocks {
        return Err(PyValueError::new_err("Fingerprint bit mismatch"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "_save_fingerprints")]
pub(crate) fn save_fingerprints<'py>(
    fps: &Bound<'py, PySequence>,
    path: &Bound<'py, PyAny>,
) -> PyResult<()> {
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
pub(crate) fn fingerprints_to_blocks<'py>(
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
pub(crate) fn load_fingerprints<'py>(
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
#[pyo3(signature = (smiles, path, radius=2, n_bits=2048, *, n_threads=-1, verbose=false))]
pub(crate) fn save_morgan_fps_from_smiles(
    py: Python<'_>,
    smiles: Vec<String>,
    path: PathBuf,
    radius: u32,
    n_bits: u32,
    n_threads: i32,
    verbose: bool,
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
    let failed: PyResult<Vec<bool>> = run_pool_verbose(
        py,
        &pool,
        verbose,
        smiles.len(),
        "Fingerprinting SMILES",
        true,
        |progress| {
            blocks
                .par_chunks_mut(n_blocks)
                .zip(smiles.par_iter())
                .map(|(out, smi)| ffi::smiles_to_fp_blocks(generator, smi, out) != 0)
                .map(|failed| progress.track(failed))
                .collect::<Result<Vec<_>, Cancelled>>()
        },
    );
    let failed = match failed {
        Ok(failed) => failed,
        Err(e) => {
            ffi::free_morgan_generator(generator);
            return Err(e);
        }
    };
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
