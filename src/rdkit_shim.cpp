#include "smiles_fp_rs/src/rdkit_shim.h"
#include <boost/python.hpp>
#include <boost/dynamic_bitset.hpp>
#include <DataStructs/ExplicitBitVect.h>
#include <fstream>

#if defined(__GNUC__) || defined(__clang__)
    #define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
    #define ALWAYS_INLINE __forceinline
#else
    #define ALWAYS_INLINE inline
#endif

namespace py = boost::python;

struct PySequenceData {
    py::object obj;       // Keeps the Python sequence alive!
    PyObject** items;     // Raw pointer array
    Py_ssize_t len;
    uint32_t n_fps;       // Number of fingerprints
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

    try {
        if (!fout.is_open()) {
            PyErr_SetString(PyExc_ValueError, "Inconsistent fingerprint lengths");
            throw py::error_already_set();
        }

        PySequenceData seq_data = extract_sequence_data(py_list_ptr);

        if (seq_data.n_fps == 0) {
            char zeros[8] = {0};
            fout.write(zeros, 8);
            fout.close();
            return;
        }

        // Extract first fingerprint to get bit length
        py::extract<ExplicitBitVect&> first_extractor(seq_data.items[0]);
        ExplicitBitVect& first_fp = first_extractor();
        uint32_t n_bits = first_fp.getNumBits();

        fout.write(reinterpret_cast<const char*>(&seq_data.n_fps), sizeof(seq_data.n_fps));
        fout.write(reinterpret_cast<const char*>(&n_bits), sizeof(n_bits));

        std::vector<boost::dynamic_bitset<>::block_type> buffer;

        // Loop and reuse buffer just like your pure C++ version
        for (Py_ssize_t i = 0; i < seq_data.len; ++i) {
            py::extract<ExplicitBitVect&> extractor(seq_data.items[i]);
            ExplicitBitVect& fp = extractor();

            if (fp.getNumBits() != n_bits) {
                PyErr_SetString(PyExc_ValueError, "Inconsistent fingerprint lengths");
                throw py::error_already_set();
            }

            buffer.clear();
            boost::to_block_range(*(fp.dp_bits), std::back_inserter(buffer));

            fout.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(buffer[0]));
        }

        fout.close();
    } catch (const py::error_already_set&) {
        PyErr_Print();
        PyErr_Clear();
    }
}

FpBatch extract_rdkit_bit_vects(size_t py_list_ptr) {
    FpBatch result;

    try {
        PySequenceData seq_data = extract_sequence_data(py_list_ptr);
        result.n_fps = seq_data.n_fps;

        for (Py_ssize_t i = 0; i < seq_data.len; ++i) {
            py::extract<ExplicitBitVect&> extractor(seq_data.items[i]);
            ExplicitBitVect& fp = extractor();

            std::vector<uint64_t> blocks;
            boost::to_block_range(*(fp.dp_bits), std::back_inserter(blocks));

            if (i == 0) {
                result.n_bits = fp.getNumBits();
            } else if (result.n_bits != fp.getNumBits()) {
                // Raise descriptive error instead of soft failing
                PyErr_SetString(PyExc_ValueError, "Inconsistent fingerprint lengths");
                throw py::error_already_set();
            }
            for(uint64_t b : blocks) result.data.push_back(b);
        }
    } catch (const py::error_already_set&) {
        PyErr_Print();
        PyErr_Clear();
        // Return empty on failure
        result.n_fps = 0;
        result.n_bits = 0;
        result.data.clear();
    }
    return result;
}

size_t create_rdkit_bit_vects(uint32_t n_fps, uint32_t n_bits, rust::Slice<const uint64_t> blocks) {
    try {
        PyObject* py_list = PyList_New(n_fps);
        uint32_t n_blocks = (n_bits + 63) / 64;

        ExplicitBitVect empty_bv(n_bits);

        for (uint32_t i = 0; i < n_fps; ++i) {
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
        PyErr_Print();
        PyErr_Clear();
        return 0;
    }
}
