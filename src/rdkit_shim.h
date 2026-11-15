#pragma once
#include <cstddef>

#include "rust/cxx.h"
#include "smiles_fp/src/lib.rs.h"

void write_rdkit_bit_vects_to_file(size_t py_list_ptr, rust::Str filename);
FpBatch extract_rdkit_bit_vects(size_t py_list_ptr);
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

// Returns an owning deep copy of a molecule without touching the Python API. The caller frees it
// with `free_mol_raw`.
size_t clone_mol_raw(size_t mol_ptr);
void free_mol_raw(size_t mol_ptr);
