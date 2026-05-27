#pragma once
#include "rust/cxx.h"
#include <cstddef>
#include "smiles_fp_rs/src/lib.rs.h"

void write_rdkit_bit_vects_to_file(size_t py_list_ptr, rust::Str filename);
FpBatch extract_rdkit_bit_vects(size_t py_list_ptr);
size_t create_rdkit_bit_vects(uint32_t n_fps, uint32_t n_bits, rust::Slice<const uint64_t> blocks);
