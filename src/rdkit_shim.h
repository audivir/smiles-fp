#pragma once
#include <cstddef>

#include "rust/cxx.h"
#include "smiles_fp/src/lib.rs.h"

// Writes a Python sequence of `ExplicitBitVect` to a fingerprint file. Requires the GIL. Removes
// the partial file and throws on failure: `PythonErrorPending` when a Python error is set (for
// example inconsistent fingerprint lengths), `std::runtime_error` for I/O errors.
void write_rdkit_bit_vects_to_file(size_t py_list_ptr, rust::Str filename);
// Copies the bits of a Python sequence of `ExplicitBitVect` into a batch of 64-bit blocks. Requires
// the GIL. Throws `PythonErrorPending` instead of returning a partial batch when an item is not a
// bit vector or the lengths differ.
FpBatch extract_rdkit_bit_vects(size_t py_list_ptr);
// Builds a new Python list of `ExplicitBitVect` from 64-bit blocks, `n_bits` bits per fingerprint.
// Requires the GIL. Returns the list as an integer, or 0 with a Python error set on failure.
size_t create_rdkit_bit_vects(uint32_t n_fps, uint32_t n_bits,
                              rust::Slice<const uint64_t> blocks);

// Parses and sanitizes SMILES like `Chem.MolFromSmiles` without touching the Python API, so it is
// thread-safe. Returns an owning `ROMol*` as an integer, or 0 if the SMILES is invalid. The caller
// passes the pointer to `wrap_mols` or `free_mol_raw`.
size_t parse_smiles_raw(rust::Str smiles, bool sanitize);
// Wraps molecules in a new Python list of `Chem.Mol`, with None for 0 pointers. Requires the GIL
// and takes ownership of every non-zero pointer. Returns 0 on failure with a Python error set.
size_t wrap_mols(rust::Slice<const size_t> ptrs);

// Collects borrowed `ROMol` pointers from a Python sequence of `Chem.Mol`. Requires the GIL.
// Returns an owning reference to the fast sequence in `keepalive`, which the caller keeps alive
// while using the pointers and then releases. Sets a Python TypeError for items that are not
// molecules.
MolPtrs extract_mol_ptrs(size_t py_seq_ptr);
// Writes SMILES with `Chem.MolToSmiles` semantics without touching the Python API, so it is
// thread-safe on distinct molecules. The `doRandom` option is unsupported because the random
// generator of RDKit is global and not thread-safe.
rust::String mol_to_smiles_raw(size_t mol_ptr, const SmilesOpts& opts);

// Creates a Morgan fingerprint generator that all threads share read-only. Returns it as an
// integer, which the caller frees with `free_morgan_generator`.
size_t make_morgan_generator(uint32_t radius, uint32_t n_bits);
void free_morgan_generator(size_t generator);
// Parses and sanitizes one SMILES, fingerprints it, and writes the 64-bit blocks of the
// fingerprint into `out` without touching the Python API, so it is thread-safe. Returns 0 on
// success and 1 if the SMILES is invalid or the fingerprint cannot be computed.
int32_t smiles_to_fp_blocks(size_t generator, rust::Str smiles, rust::Slice<uint64_t> out);

// Returns an owning deep copy of a molecule without touching the Python API. The caller frees it
// with `free_mol_raw`.
size_t clone_mol_raw(size_t mol_ptr);
// Frees a molecule returned by `parse_smiles_raw` or `clone_mol_raw` that was not wrapped.
void free_mol_raw(size_t mol_ptr);
