#include "smiles_fp/src/rdkit_shim.h"

#include <DataStructs/ExplicitBitVect.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/Fingerprints/MorganGenerator.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolStandardize/Charge.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include <boost/dynamic_bitset.hpp>
#include <boost/python.hpp>
#include <bitset>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
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
  const std::string smiles = RDKit::MolToSmiles(
      *mol, opts.isomeric_smiles, opts.kekule_smiles, opts.rooted_at_atom,
      opts.canonical, opts.all_bonds_explicit, opts.all_hs_explicit,
      /*doRandom=*/false, opts.ignore_atom_map_numbers);
  // the pointer and length constructor avoids `std::string` in the signature, which would not link
  // with the old libstdc++ string ABI of some RDKit wheels.
  return rust::String(smiles.data(), smiles.size());
}

size_t clone_mol_raw(size_t mol_ptr) {
  const RDKit::ROMol* mol = reinterpret_cast<const RDKit::ROMol*>(mol_ptr);
  return reinterpret_cast<size_t>(new RDKit::ROMol(*mol));
}

void free_mol_raw(size_t mol_ptr) {
  delete reinterpret_cast<RDKit::ROMol*>(mol_ptr);
}

size_t make_morgan_generator(uint32_t radius, uint32_t n_bits) {
  // these options and the default atom invariants match `GetMorganGenerator` in Python.
  return reinterpret_cast<size_t>(
      RDKit::MorganFingerprint::getMorganGenerator<std::uint64_t>(
          radius, /*countSimulation=*/false, /*includeChirality=*/false,
          /*useBondTypes=*/true, /*onlyNonzeroInvariants=*/false,
          /*atomInvariantsGenerator=*/nullptr, /*bondInvariantsGenerator=*/nullptr,
          n_bits));
}

void free_morgan_generator(size_t generator) {
  delete reinterpret_cast<RDKit::FingerprintGenerator<std::uint64_t>*>(generator);
}

int32_t smiles_to_fp_blocks(size_t generator, rust::Str smiles, rust::Slice<uint64_t> out) {
  try {
    std::string smi(smiles.data(), smiles.size());
    std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smi));
    if (!mol) return 1;
    auto* gen = reinterpret_cast<const RDKit::FingerprintGenerator<std::uint64_t>*>(generator);
    std::unique_ptr<ExplicitBitVect> fp(gen->getFingerprint(*mol));
    std::vector<boost::dynamic_bitset<>::block_type> blocks;
    boost::to_block_range(*(fp->dp_bits), std::back_inserter(blocks));
    if (blocks.size() != out.size()) return 1;
    std::copy(blocks.begin(), blocks.end(), out.begin());
    return 0;
  } catch (...) {
    return 1;
  }
}

struct SaltStripper {
  std::vector<std::unique_ptr<RDKit::RWMol>> salts;
};

static SaltStripper build_stripper(rust::Str smarts_lines) {
  SaltStripper stripper;
  std::istringstream in(std::string(smarts_lines.data(), smarts_lines.size()));
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::unique_ptr<RDKit::RWMol> query(RDKit::SmartsToMol(line));
    if (!query) throw std::runtime_error("Invalid salt SMARTS: " + line);
    stripper.salts.push_back(std::move(query));
  }
  return stripper;
}

size_t make_salt_stripper(rust::Str smarts_lines) {
  return reinterpret_cast<size_t>(new SaltStripper(build_stripper(smarts_lines)));
}

void free_salt_stripper(size_t stripper) {
  delete reinterpret_cast<SaltStripper*>(stripper);
}

// deletes fragments that match `salt` until the atom count stops shrinking.
static void delete_salt(std::unique_ptr<RDKit::ROMol>& mol, const RDKit::ROMol& salt,
                        bool keep_last) {
  size_t n_atoms = mol->getNumAtoms();
  if (!n_atoms) return;
  std::unique_ptr<RDKit::ROMol> next(RDKit::deleteSubstructs(*mol, salt, true));
  if (keep_last && next->getNumAtoms() == 0) return;
  mol = std::move(next);
  while (mol->getNumAtoms() && n_atoms > mol->getNumAtoms()) {
    n_atoms = mol->getNumAtoms();
    next.reset(RDKit::deleteSubstructs(*mol, salt, true));
    if (keep_last && next->getNumAtoms() == 0) break;
    mol = std::move(next);
  }
}

// strips a copy of `in` like `SaltRemover.StripMol`. Throws if sanitizing the result fails.
static std::unique_ptr<RDKit::RWMol> strip_salts(const SaltStripper& stripper,
                                                 const RDKit::ROMol& in, bool keep_last) {
  std::vector<int> mapping;
  auto single_fragment = [&](const RDKit::ROMol& m) {
    mapping.clear();
    return RDKit::MolOps::getMolFrags(m, mapping) <= 1;
  };
  auto mol = std::make_unique<RDKit::ROMol>(in);
  bool modified = false;
  if (!(keep_last && single_fragment(*mol))) {
    size_t n_atoms = mol->getNumAtoms();
    for (const auto& salt : stripper.salts) {
      delete_salt(mol, *salt, keep_last);
      if (n_atoms == mol->getNumAtoms()) continue;
      n_atoms = mol->getNumAtoms();
      modified = true;
      if (keep_last && single_fragment(*mol)) break;
    }
  }
  auto result = std::make_unique<RDKit::RWMol>(*mol);
  if (modified && result->getNumAtoms() > 0) RDKit::MolOps::sanitizeMol(*result);
  return result;
}

size_t strip_salts_raw(size_t stripper, size_t mol_ptr, bool keep_last) {
  try {
    auto result = strip_salts(*reinterpret_cast<const SaltStripper*>(stripper),
                              *reinterpret_cast<const RDKit::ROMol*>(mol_ptr), keep_last);
    return reinterpret_cast<size_t>(static_cast<RDKit::ROMol*>(result.release()));
  } catch (...) {
    return 0;
  }
}

struct Cleaner {
  CleanOpts opts;
  std::bitset<256> allowed;  // empty means every element is allowed.
  SaltStripper salts;
  RDKit::MolStandardize::Uncharger uncharger;
};

size_t make_cleaner(const CleanOpts& opts, rust::Slice<const uint32_t> allowed,
                    rust::Str salt_lines) {
  auto cleaner = std::make_unique<Cleaner>();
  cleaner->opts = opts;
  // `MolToSmiles` drops isotopes and stereochemistry itself when not isomeric.
  cleaner->opts.no_isotopes = opts.isomeric && opts.no_isotopes;
  cleaner->opts.no_stereo = opts.isomeric && opts.no_stereo;
  for (uint32_t z : allowed) {
    if (z < cleaner->allowed.size()) cleaner->allowed.set(z);
  }
  if (opts.no_salts) cleaner->salts = build_stripper(salt_lines);
  return reinterpret_cast<size_t>(cleaner.release());
}

void free_cleaner(size_t cleaner) {
  delete reinterpret_cast<Cleaner*>(cleaner);
}

// cleans `mol` in place and returns whether it is still valid.
static bool clean_mol(Cleaner& c, std::unique_ptr<RDKit::RWMol>& mol) {
  if (mol->getNumAtoms() == 0) return false;
  RDKit::MolOps::removeAllHs(*mol);
  if (mol->getNumAtoms() == 0) return false;
  if (c.opts.no_salts) {
    mol = strip_salts(c.salts, *mol, false);
    if (mol->getNumAtoms() == 0) return false;
  }
  if (c.opts.neutralize) c.uncharger.unchargeInPlace(*mol);
  if (c.opts.no_isotopes) {
    for (auto* atom : mol->atoms()) atom->setIsotope(0);
  }
  if (c.opts.no_stereo) RDKit::MolOps::removeStereochemistry(*mol);
  if (c.allowed.any()) {
    for (const auto* atom : mol->atoms()) {
      auto z = static_cast<size_t>(atom->getAtomicNum());
      if (z >= c.allowed.size() || !c.allowed[z]) return false;
    }
  }
  return mol->getNumAtoms() > 0;
}

// writes the SMILES of a cleaned molecule and, with `verify`, checks that it parses again. Returns
// an empty SMILES otherwise. With `want_mol`, `mol` owns the molecule parsed from the written SMILES.
static Cleaned write_checked(const Cleaner& c, const RDKit::ROMol& mol, bool want_mol) {
  Cleaned result{rust::String(), 0};
  const std::string out = RDKit::MolToSmiles(mol, c.opts.isomeric);
  if (!want_mol && !c.opts.verify) {
    if (!out.empty()) result.smiles = rust::String(out.data(), out.size());
    return result;
  }
  std::unique_ptr<RDKit::RWMol> check(RDKit::SmilesToMol(out));
  if (out.empty() || !check || check->getNumAtoms() == 0) return result;
  result.smiles = rust::String(out.data(), out.size());
  if (want_mol) {
    result.mol = reinterpret_cast<size_t>(static_cast<RDKit::ROMol*>(check.release()));
  }
  return result;
}

Cleaned clean_smiles_raw(size_t cleaner, rust::Str smiles, bool want_mol) {
  try {
    if (smiles.size() == 0) return Cleaned{rust::String(), 0};
    auto* c = reinterpret_cast<Cleaner*>(cleaner);
    std::unique_ptr<RDKit::RWMol> mol(
        RDKit::SmilesToMol(std::string(smiles.data(), smiles.size())));
    if (!mol || !clean_mol(*c, mol)) return Cleaned{rust::String(), 0};
    return write_checked(*c, *mol, want_mol);
  } catch (...) {
    return Cleaned{rust::String(), 0};
  }
}

size_t clone_rwmol_raw(size_t mol_ptr) {
  const auto* mol = reinterpret_cast<const RDKit::ROMol*>(mol_ptr);
  return reinterpret_cast<size_t>(static_cast<RDKit::ROMol*>(new RDKit::RWMol(*mol)));
}

Cleaned clean_mol_raw(size_t cleaner, size_t mol_ptr, bool want_smiles) {
  // takes ownership, so the copy is freed on every path.
  std::unique_ptr<RDKit::RWMol> mol(
      static_cast<RDKit::RWMol*>(reinterpret_cast<RDKit::ROMol*>(mol_ptr)));
  try {
    auto* c = reinterpret_cast<Cleaner*>(cleaner);
    if (!clean_mol(*c, mol)) return Cleaned{rust::String(), 0};
    if (want_smiles) return write_checked(*c, *mol, false);
    return Cleaned{rust::String(), reinterpret_cast<size_t>(static_cast<RDKit::ROMol*>(mol.release()))};
  } catch (...) {
    return Cleaned{rust::String(), 0};
  }
}

bool clean_mol_inplace_raw(size_t cleaner, size_t mol_ptr) {
  try {
    auto* c = reinterpret_cast<Cleaner*>(cleaner);
    auto* orig = reinterpret_cast<RDKit::ROMol*>(mol_ptr);
    // works on a private copy, so a molecule that turns out invalid stays unchanged.
    auto mol = std::make_unique<RDKit::RWMol>(*orig);
    if (!clean_mol(*c, mol)) return false;
    // `RWMol` adds no members to `ROMol`, and the in-place functions of RDKit cast the same way.
    // `clear` frees the old atoms and bonds, which the move assignment would leak.
    auto& target = static_cast<RDKit::RWMol&>(*orig);
    target.clear();
    target = std::move(*mol);
    return true;
  } catch (...) {
    return false;
  }
}
