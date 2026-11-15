#include "smiles_fp/src/rdkit_shim.h"

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include <boost/dynamic_bitset.hpp>
#include <boost/python.hpp>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif

namespace py = boost::python;

// A Python exception is pending and deliberately not cleared. The Rust side fetches it with
// `PyErr::take`, so the caller sees the original exception type and message.
struct PythonErrorPending : std::runtime_error {
  PythonErrorPending() : std::runtime_error("Python exception pending") {}
};

struct PySequenceData {
  py::object obj;    // Keeps the Python sequence alive!
  PyObject** items;  // Raw pointer array
  Py_ssize_t len;
  uint32_t n_fps;  // Number of fingerprints
};

ALWAYS_INLINE PySequenceData extract_sequence_data(size_t py_list_ptr) {
  PyObject* seq_ptr = reinterpret_cast<PyObject*>(py_list_ptr);

  // Get the fast sequence
  PyObject* fast_seq = PySequence_Fast(seq_ptr, "Expected a sequence");
  if (!fast_seq) {
    throw py::error_already_set();
  }

  PySequenceData data;
  // Wrap in py::object (using handle to consume the reference count).
  // This ensures the memory isn't freed until 'data.obj' goes out of scope.
  data.obj = py::object(py::handle<>(fast_seq));
  data.items = PySequence_Fast_ITEMS(fast_seq);
  data.len = PySequence_Fast_GET_SIZE(fast_seq);
  data.n_fps = static_cast<uint32_t>(data.len);

  return data;
}

void write_rdkit_bit_vects_to_file(size_t py_list_ptr, rust::Str filename) {
  std::string fname(filename.data(), filename.size());
  std::ofstream fout(fname, std::ios::binary);
  if (!fout.is_open()) {
    throw std::runtime_error("Could not open '" + fname + "' for writing");
  }

  // removes the partial file on failure, so a truncated file never looks valid.
  auto discard_partial_file = [&]() {
    fout.close();
    std::remove(fname.c_str());
  };

  try {
    PySequenceData seq_data = extract_sequence_data(py_list_ptr);

    if (seq_data.len > static_cast<Py_ssize_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("Too many fingerprints for the file format (max 2^32-1)");
    }

    if (seq_data.n_fps == 0) {
      char zeros[8] = {0};
      fout.write(zeros, 8);
    } else {
      // Extract first fingerprint to get bit length
      py::extract<ExplicitBitVect&> first_extractor(seq_data.items[0]);
      ExplicitBitVect& first_fp = first_extractor();
      uint32_t n_bits = first_fp.getNumBits();

      fout.write(reinterpret_cast<const char*>(&seq_data.n_fps),
                 sizeof(seq_data.n_fps));
      fout.write(reinterpret_cast<const char*>(&n_bits), sizeof(n_bits));

      std::vector<boost::dynamic_bitset<>::block_type> buffer;

      for (Py_ssize_t i = 0; i < seq_data.len; ++i) {
        py::extract<ExplicitBitVect&> extractor(seq_data.items[i]);
        ExplicitBitVect& fp = extractor();

        if (fp.getNumBits() != n_bits) {
          PyErr_SetString(PyExc_ValueError, "Inconsistent fingerprint lengths");
          throw py::error_already_set();
        }

        buffer.clear();
        boost::to_block_range(*(fp.dp_bits), std::back_inserter(buffer));

        fout.write(reinterpret_cast<const char*>(buffer.data()),
                   buffer.size() * sizeof(buffer[0]));
      }
    }

    fout.close();
    if (fout.fail()) {
      throw std::runtime_error("Failed writing '" + fname + "' (disk full?)");
    }
  } catch (const py::error_already_set&) {
    discard_partial_file();
    throw PythonErrorPending();
  } catch (...) {
    discard_partial_file();
    throw;
  }
}

FpBatch extract_rdkit_bit_vects(size_t py_list_ptr) {
  FpBatch result;
  result.n_fps = 0;
  result.n_bits = 0;

  try {
    PySequenceData seq_data = extract_sequence_data(py_list_ptr);
    if (seq_data.len > static_cast<Py_ssize_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("Too many fingerprints (max 2^32-1)");
    }
    result.n_fps = seq_data.n_fps;

    for (Py_ssize_t i = 0; i < seq_data.len; ++i) {
      py::extract<ExplicitBitVect&> extractor(seq_data.items[i]);
      ExplicitBitVect& fp = extractor();

      std::vector<uint64_t> blocks;
      boost::to_block_range(*(fp.dp_bits), std::back_inserter(blocks));

      if (i == 0) {
        result.n_bits = fp.getNumBits();
      } else if (result.n_bits != fp.getNumBits()) {
        PyErr_SetString(PyExc_ValueError, "Inconsistent fingerprint lengths");
        throw py::error_already_set();
      }
      for (uint64_t b : blocks) result.data.push_back(b);
    }
  } catch (const py::error_already_set&) {
    // rethrows instead of returning an empty batch, which callers would read as no fingerprints.
    throw PythonErrorPending();
  }
  return result;
}

size_t create_rdkit_bit_vects(uint32_t n_fps, uint32_t n_bits,
                              rust::Slice<const uint64_t> blocks) {
  PyObject* py_list = PyList_New(n_fps);
  if (!py_list) return 0;  // MemoryError is already set

  try {
    const size_t n_blocks = (static_cast<size_t>(n_bits) + 63) / 64;

    ExplicitBitVect empty_bv(n_bits);

    for (size_t i = 0; i < n_fps; ++i) {
      auto start = blocks.begin() + (i * n_blocks);

      ExplicitBitVect fp(n_bits);
      boost::from_block_range(start, start + n_blocks, *(fp.dp_bits));
      fp.dp_bits->resize(n_bits);

      fp |= empty_bv;

      py::object py_fp(fp);
      PyObject* ptr = py_fp.ptr();
      Py_INCREF(ptr);
      PyList_SET_ITEM(py_list, i, ptr);
    }
    return reinterpret_cast<size_t>(py_list);
  } catch (const py::error_already_set&) {
    // keeps the Python error set for the caller and frees the half-built list.
    Py_DECREF(py_list);
    return 0;
  } catch (const std::exception& e) {
    Py_DECREF(py_list);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
}

size_t parse_smiles_raw(rust::Str smiles, bool sanitize) {
  try {
    std::string smi(smiles.data(), smiles.size());
    RDKit::RWMol* m = RDKit::SmilesToMol(smi, 0, sanitize);
    return reinterpret_cast<size_t>(static_cast<RDKit::ROMol*>(m));
  } catch (...) {
    return 0;
  }
}

size_t wrap_mols(rust::Slice<const size_t> ptrs) {
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(ptrs.size()));
  if (!list) return 0;
  size_t i = 0;
  try {
    for (; i < ptrs.size(); ++i) {
      if (ptrs[i] == 0) {
        Py_INCREF(Py_None);
        PyList_SET_ITEM(list, i, Py_None);
        continue;
      }
      // `ROMol` is noncopyable and registered with a `shared_ptr` holder, so Python co-owns the
      // same object without a copy.
      RDKit::ROMOL_SPTR sp(reinterpret_cast<RDKit::ROMol*>(ptrs[i]));
      py::object o(sp);
      PyObject* p = o.ptr();
      Py_INCREF(p);
      PyList_SET_ITEM(list, i, p);
    }
    return reinterpret_cast<size_t>(list);
  } catch (const py::error_already_set&) {
    // keeps the Python error set and frees the molecules that were not wrapped yet.
    for (size_t j = i + 1; j < ptrs.size(); ++j)  // ptrs[i] was adopted by sp
      delete reinterpret_cast<RDKit::ROMol*>(ptrs[j]);
    Py_DECREF(list);
    return 0;
  }
}

MolPtrs extract_mol_ptrs(size_t py_seq_ptr) {
  PyObject* fast = PySequence_Fast(reinterpret_cast<PyObject*>(py_seq_ptr),
                                   "Expected a sequence of molecules");
  if (!fast) throw PythonErrorPending();

  MolPtrs result;
  result.keepalive = reinterpret_cast<size_t>(fast);
  try {
    PyObject** items = PySequence_Fast_ITEMS(fast);
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    result.ptrs.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      py::extract<RDKit::ROMol&> extractor(items[i]);
      result.ptrs.push_back(reinterpret_cast<size_t>(&extractor()));
    }
  } catch (const py::error_already_set&) {
    Py_DECREF(fast);
    throw PythonErrorPending();
  }
  return result;
}

rust::String mol_to_smiles_raw(size_t mol_ptr, const SmilesOpts& opts) {
  const RDKit::ROMol* mol = reinterpret_cast<const RDKit::ROMol*>(mol_ptr);
  return rust::String(RDKit::MolToSmiles(
      *mol, opts.isomeric_smiles, opts.kekule_smiles, opts.rooted_at_atom,
      opts.canonical, opts.all_bonds_explicit, opts.all_hs_explicit,
      /*doRandom=*/false, opts.ignore_atom_map_numbers));
}

size_t clone_mol_raw(size_t mol_ptr) {
  const RDKit::ROMol* mol = reinterpret_cast<const RDKit::ROMol*>(mol_ptr);
  return reinterpret_cast<size_t>(new RDKit::ROMol(*mol));
}

void free_mol_raw(size_t mol_ptr) {
  delete reinterpret_cast<RDKit::ROMol*>(mol_ptr);
}
