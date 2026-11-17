use pyo3::{
    exceptions::{PyOSError, PyRuntimeError},
    prelude::*,
    types::PyModule,
};

mod clean;
mod fps;
mod mols;
mod murcko;
mod pool;
mod salt_remover;
mod similarity;

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
        largest_fragment: bool,
        no_salts: bool,
        no_isotopes: bool,
        no_stereo: bool,
        isomeric: bool,
        neutralize: bool,
        verify: bool,
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
        fn clean_mol_inplace_raw(cleaner: usize, mol_ptr: usize) -> bool;
        fn murcko_raw(mol_ptr: usize) -> usize;
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
pub(crate) fn ffi_error(py: Python<'_>, e: cxx::Exception, os_error: bool) -> PyErr {
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

#[pymodule]
fn _smiles_fp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fps::save_fingerprints, m)?)?;
    m.add_function(wrap_pyfunction!(fps::load_fingerprints, m)?)?;
    m.add_function(wrap_pyfunction!(fps::fingerprints_to_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(mols::mols_from_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(mols::mols_to_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(salt_remover::strip_salts, m)?)?;
    m.add_function(wrap_pyfunction!(clean::clean, m)?)?;
    m.add_function(wrap_pyfunction!(murcko::murcko_scaffolds, m)?)?;
    m.add_function(wrap_pyfunction!(fps::save_morgan_fps_from_smiles, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::bulk_tanimoto_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(
        similarity::bulk_tanimoto_parallel_topk,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(similarity::bulk_tanimoto_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::bulk_tanimoto_mmap_topk, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::internal_tanimoto_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(similarity::internal_tanimoto_mmap, m)?)?;
    Ok(())
}
