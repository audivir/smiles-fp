#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/object.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <DataStructs/ExplicitBitVect.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

// It is crucial to include the NumPy headers for the C-API
#define PY_ARRAY_UNIQUE_SYMBOL rdkit_bulk_tanimoto_ARRAY_API
#include <numpy/arrayobject.h>

namespace py = boost::python;

// Helper function to popcount a single 64-bit block. This is a critical
// performance primitive. Modern compilers will turn this into a single CPU
// instruction.
inline int popcount_block(uint64_t block) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(block);
#elif defined(_MSC_VER)
  return static_cast<int>(__popcnt64(block));
#else
  // A reasonably fast fallback for other compilers
  block -= (block >> 1) & 0x5555555555555555ULL;
  block =
      (block & 0x3333333333333333ULL) + ((block >> 2) & 0x3333333333333333ULL);
  block = (block + (block >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
  return static_cast<int>((block * 0x0101010101010101ULL) >> 56);
#endif
}

/**
 * Converts Python StrPath (str or pathlib.Path) to std::string.
 * Strictly rejects 'bytes' objects.
 * Raises ValueError on empty strings.
 */
std::string pathlike_to_string(py::object py_path) {
    PyObject* obj = py_path.ptr();

    // 1. Handle pathlib.Path / os.PathLike
    if (PyObject_HasAttrString(obj, "__fspath__")) {
        py::handle<> path_repr(PyObject_CallMethod(obj, "__fspath__", nullptr));
        obj = path_repr.get();
    }

    // 2. Strict type check: Only allow Unicode (str)
    if (!PyUnicode_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a str or an os.PathLike returning a str (bytes not accepted).");
        py::throw_error_already_set();
    }

    // 3. Extract and check for empty string
    const char* c_str = PyUnicode_AsUTF8(obj);
    if (!c_str || c_str[0] == '\0') {
        PyErr_SetString(PyExc_ValueError, "Filename cannot be empty.");
        py::throw_error_already_set();
    }

    return std::string(c_str);
}

/*
Call this in long-running loops to check for Python interrupts. This is
necessary to be able to ctrl-c out of Python scripts that are running C++ code.
*/
void check_for_interrupt() {
  if (PyErr_CheckSignals() != 0) {
    py::throw_error_already_set();
  }
}

template <typename T>
void check_for_ioerror(const T &stream, const std::string &filename) {
  static_assert(std::is_base_of<std::ios_base, T>::value, 
                "check_for_ioerror: T must be a stream type (e.g., std::ifstream, std::ofstream)");
  if (!stream.is_open()) {
    PyErr_SetString(PyExc_IOError,
                    ("Could not open file: " + filename).c_str());
    py::throw_error_already_set();
  }
}

void check_for_sequence(const py::object &py_seq) {
  PyObject *p = py_seq.ptr();
  if (!PySequence_Check(p) || PyBytes_Check(p) || PyUnicode_Check(p)) {
    PyErr_SetString(PyExc_TypeError, "Expected a sequence of objects.");
    py::throw_error_already_set();
  }
}

/*
Get the number of threads to use for parallel processing
If negative, use all available threads, don't use more than the number of cores
*/
unsigned int get_num_threads(int &num_threads) {
  unsigned int max_threads = std::thread::hardware_concurrency();
  if (num_threads <= 0) {
    num_threads = std::max(1u, max_threads);
  }
  unsigned int u_threads = std::min((unsigned int)num_threads, max_threads);
  return u_threads;
}

std::vector<ExplicitBitVect *>
extract_fps_from_pyseq(const py::object &py_seq) {
  check_for_sequence(py_seq);

  PyObject *seq_ptr = py_seq.ptr();
  ssize_t len = PySequence_Size(seq_ptr);
  if (len < 0) { // PySequence_Size can return -1 on error
    py::throw_error_already_set();
  }
  if (len == 0) {
    return {};
  }

  std::vector<ExplicitBitVect *> result;
  result.reserve(len);

  // Get C++ type information for ExplicitBitVect once
  const py::type_info cpp_type =
      py::type_id<ExplicitBitVect>();
  const py::converter::registration *reg =
      py::converter::registry::query(cpp_type);

  if (!reg || !reg->m_class_object) {
    PyErr_SetString(
        PyExc_TypeError,
        "ExplicitBitVect type is not registered with Boost.Python. Is is imported?");
    py::throw_error_already_set();
  }
  PyTypeObject *expected_py_type = reg->m_class_object;

  // Use PySequence_Fast to get a C-style array of PyObject*, which is very fast
  // and handles lists, tuples, etc. efficiently. It also manages reference
  // counts.
  py::handle<> fast_seq(
      PySequence_Fast(seq_ptr, "Expected a sequence"));
  if (!fast_seq) {
    py::throw_error_already_set();
  }

  PyObject **items = PySequence_Fast_ITEMS(fast_seq.get());
  for (ssize_t i = 0; i < len; ++i) {
    PyObject *item = items[i];

    // 1. Perform a fast type-check
    if (Py_TYPE(item) != expected_py_type) {
      PyErr_SetString(PyExc_TypeError,
                      "Sequence contains a non-ExplicitBitVect object.");
      py::throw_error_already_set();
    }

    // 2. Safely find the stored C++ pointer using Boost.Python's internal
    // helper
    py::extract<ExplicitBitVect *> extractor(item);

    // 3. The helper returns a pointer to the stored pointer (a T** for a held
    // type T*). So we cast to a pointer-to-pointer and dereference it once.
    result.push_back(extractor());
  }

  return result;
}

// Tanimoto calculation remains the same
double tanimoto_similarity(const ExplicitBitVect &bv1,
                           const ExplicitBitVect &bv2) {
  unsigned int total = bv1.getNumOnBits() + bv2.getNumOnBits();
  if (total == 0) {
    return 0.0; // by definition
  }
  boost::dynamic_bitset<> intersect =
      (*bv1.dp_bits) & (*bv2.dp_bits);
  unsigned int common = static_cast<unsigned int>(intersect.count());
  if (common == total) {
    return 1.0; // by definition
  }
  return static_cast<double>(common) / (total - common);
}

// This worker function writes directly into a slice of a pre-allocated vector
void bulk_tanimoto_worker(const std::vector<ExplicitBitVect *> &chunk,
                          const std::vector<ExplicitBitVect *> &fps2,
                          double *results_ptr) {

  size_t i = 0;
  for (const auto &bv1 : chunk) {
    for (const auto &bv2 : fps2) {
      results_ptr[i++] = tanimoto_similarity(*bv1, *bv2);
    }
  }
}

// The main parallel function, returning a NumPy array
py::object
bulk_tanimoto_parallel(const py::object &py_fps,
                       const py::object &py_fps2,
                       int num_threads = -1) {
  // This needs to be called once to initialize the NumPy C-API
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    py::throw_error_already_set();
  }

  std::vector<ExplicitBitVect *> fps = extract_fps_from_pyseq(py_fps);
  std::vector<ExplicitBitVect *> fps2 = extract_fps_from_pyseq(py_fps2);

  unsigned int u_threads = get_num_threads(num_threads);

  // Pre-allocate a single flat vector for all results.
  size_t total_size = fps.size() * fps2.size();
  if (total_size == 0) {
    // Return an empty NumPy array
    npy_intp dims[1] = {0};
    PyObject *py_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    return py::object(py::handle<>(py_obj));
  }
  std::vector<double> all_results(total_size);

  size_t chunk_size = (fps.size() + u_threads - 1) / u_threads;
  std::vector<std::thread> threads;

  for (unsigned int t = 0; t < u_threads; ++t) {
    size_t start = t * chunk_size;
    if (start >= fps.size())
      continue; // Handle cases where num_threads > fps.size()
    size_t end = std::min(start + chunk_size, fps.size());

    // Create a view of the chunk for the thread
    std::vector<ExplicitBitVect *> chunk(fps.begin() + start,
                                         fps.begin() + end);

    // Calculate the pointer to the start of this thread's result slice
    double *results_ptr = &all_results[start * fps2.size()];

    threads.emplace_back(bulk_tanimoto_worker, std::move(chunk), std::ref(fps2),
                         results_ptr);
  }

  // wait for threads to finish
  for (auto &thread : threads) {
    thread.join();
  }

  // Create a 1D NumPy array.
  npy_intp dims[1] = {static_cast<npy_intp>(total_size)};
  PyObject *py_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  // Get a pointer to the NumPy array's data buffer
  void *py_arr_data = PyArray_DATA(reinterpret_cast<PyArrayObject *>(py_obj));

  // Copy all the C++ results into the NumPy array in one go
  memcpy(py_arr_data, all_results.data(), total_size * sizeof(double));

  // Wrap the PyObject* in a py::object and return it.
  // The handle<> manages the reference count correctly.
  return py::object(py::handle<>(py_obj));
}

// Helper implementation for saving fingerprints.
void save_fingerprints_impl(const py::object &py_seq,
                            const py::object &py_filename) {
  // Extract all C++ pointers at once, outside the main loop.
  std::vector<ExplicitBitVect *> fps = extract_fps_from_pyseq(py_seq);

  std::string filename = pathlike_to_string(py_filename);

  std::ofstream fout(filename, std::ios::binary);
  check_for_ioerror(fout, filename);

  uint32_t num_fps = static_cast<uint32_t>(fps.size());
  if (num_fps == 0) {
    fout.close();
    return;
  }

  // TODO: maybe include a magic number to detect wrong file format?

  // Write header: number of fingerprints and number of bits.
  fout.write(reinterpret_cast<const char *>(&num_fps), sizeof(num_fps));

  unsigned int num_bits = fps[0]->getNumBits();
  fout.write(reinterpret_cast<const char *>(&num_bits), sizeof(num_bits));

  // Reuse a single buffer to avoid repeated memory allocations.
  std::vector<boost::dynamic_bitset<>::block_type> buffer;

  // Now loop over the fast C++ vector.
  for (const auto &fp : fps) {
    check_for_interrupt(); // TODO: do we need this?

    if (fp->getNumBits() != num_bits) {
      throw std::runtime_error(
          "All fingerprints must have the same number of bits.");
    }

    const auto &bits = *(fp->dp_bits);

    // Clear and reuse the buffer, which is much cheaper than reallocating.
    buffer.clear();
    boost::to_block_range(bits, std::back_inserter(buffer));

    fout.write(reinterpret_cast<const char *>(buffer.data()),
               buffer.size() * sizeof(buffer[0]));
  }

  fout.close();
}
// Helper implementation for loading fingerprints.
py::list
load_fingerprints_impl(const py::object &py_filename) {
  std::string filename = pathlike_to_string(py_filename);

  std::ifstream fin(filename, std::ios::binary);
  check_for_ioerror(fin, filename);

  if (fin.peek() == EOF) { // Handle empty file
    return py::list();
  }

  // Read header
  uint32_t num_fps = 0, num_bits = 0;
  fin.read(reinterpret_cast<char *>(&num_fps), sizeof(num_fps));
  fin.read(reinterpret_cast<char *>(&num_bits), sizeof(num_bits));

  if (!fin) { // Handle file that only contains a partial header
    return py::list();
  }

  // We work on a vector first, then convert to a list at the end.
  std::vector<ExplicitBitVect *> fps_vec;
  fps_vec.reserve(num_fps);

  // Creating popcount helper fp once outside the loop
  auto *empty_fp_for_popcount = new ExplicitBitVect(num_bits);

  constexpr unsigned int bits_per_block =
      boost::dynamic_bitset<>::bits_per_block;
  const unsigned int num_blocks =
      (num_bits + bits_per_block - 1) / bits_per_block;
  std::vector<boost::dynamic_bitset<>::block_type> buffer(num_blocks);

  for (uint32_t i = 0; i < num_fps; ++i) {
    check_for_interrupt();

    fin.read(reinterpret_cast<char *>(buffer.data()),
             num_blocks * sizeof(buffer[0]));

    if (!fin) {
      throw std::runtime_error("File is truncated or corrupted.");
    }

    auto *fp = new ExplicitBitVect(num_bits);
    boost::from_block_range(buffer.begin(), buffer.end(), *(fp->dp_bits));

    // TODO: can we circumvent this bottleneck, we need it to update d_OnBits
    *fp = (*fp) | (*empty_fp_for_popcount);

    fps_vec.push_back(fp);
  }
  delete empty_fp_for_popcount; // Clean up the helper fp
  fin.close();

  // Create the Python list at the end, in one go.
  py::list result;
  for (const auto &fp : fps_vec) {
    result.append(py::object(py::ptr(fp)));
  }

  return result;
}

// A RAII struct to manage memory-mapped data for a fingerprint file.
struct FingerprintData {
  boost::interprocess::file_mapping mapping;
  boost::interprocess::mapped_region region;

  const uint32_t num_fps;
  const uint32_t num_bits;
  const uint32_t num_blocks;
  const uint64_t *blocks_ptr;

  FingerprintData(const std::string &filename)
      : mapping(filename.c_str(), boost::interprocess::read_only),
        region(mapping, boost::interprocess::read_only),
        num_fps(*reinterpret_cast<const uint32_t *>(
            static_cast<const char *>(region.get_address()))),
        num_bits(*reinterpret_cast<const uint32_t *>(
            static_cast<const char *>(region.get_address()) +
            sizeof(uint32_t))),
        num_blocks([this]() {
          constexpr unsigned int bits_per_block = sizeof(uint64_t) * 8;
          return (this->num_bits + bits_per_block - 1) / bits_per_block;
        }()),
        blocks_ptr(reinterpret_cast<const uint64_t *>(
            static_cast<const char *>(region.get_address()) +
            2 * sizeof(uint32_t))) {
    // Constructor body is empty, all work is done in the initializer list.
    // The lambda for num_blocks allows initialization based on another member.
  }
};

// The worker function that each thread will execute.
void tanimoto_mmap_worker(
    const FingerprintData &data1, const FingerprintData &data2,
    const std::vector<uint32_t> &popcounts2, // Pre-calculated for efficiency
    size_t start_idx1, size_t end_idx1,
    double *results_output_ptr) { // Pointer to the start of this thread's
  // output slice

  size_t current_row_offset = 0;
  for (size_t i = start_idx1; i < end_idx1; ++i) {
    // Pointer to the start of the i-th fingerprint in file 1
    const uint64_t *fp1_blocks = data1.blocks_ptr + (i * data1.num_blocks);

    // Calculate popcount for this specific fp1 on the fly
    uint32_t popcount1 = 0;
    for (uint32_t b = 0; b < data1.num_blocks; ++b) {
      popcount1 += popcount_block(fp1_blocks[b]);
    }

    // Compare against all fingerprints in file 2
    for (size_t j = 0; j < data2.num_fps; ++j) {
      const uint64_t *fp2_blocks = data2.blocks_ptr + (j * data2.num_blocks);

      // Calculate common bits
      uint32_t common_bits = 0;
      for (uint32_t b = 0; b < data1.num_blocks; ++b) {
        common_bits += popcount_block(fp1_blocks[b] & fp2_blocks[b]);
      }

      const uint32_t popcount2 = popcounts2[j];
      const uint32_t total_bits = popcount1 + popcount2;

      double *current_result = results_output_ptr + current_row_offset + j;
      if (total_bits == 0) {
        *current_result = 1.0;
      } else {
        *current_result =
            static_cast<double>(common_bits) / (total_bits - common_bits);
      }
    }
    current_row_offset += data2.num_fps;
  }
}

// The main function exposed to Python.
py::object
bulk_tanimoto_mmap(const py::object &py_filename1,
                   const py::object &py_filename2,
                   int num_threads = -1) {
  // This must be called once per function that uses the NumPy C-API
  if (_import_array() < 0) {
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    py::throw_error_already_set();
  }

  std::string filename1 = pathlike_to_string(py_filename1);
  std::string filename2 = pathlike_to_string(py_filename2);

  // 1. Memory-map files. This is RAII, so they are unmapped on scope exit.
  FingerprintData data1(filename1);
  FingerprintData data2(filename2);

  if (data1.num_bits != data2.num_bits) {
    throw std::runtime_error(
        "Fingerprint bit counts in both files must match.");
  }

  // 2. Pre-compute popcounts for the second file (the inner loop)
  std::vector<uint32_t> popcounts2(data2.num_fps);
  for (uint32_t i = 0; i < data2.num_fps; ++i) {
    const uint64_t *fp_blocks = data2.blocks_ptr + (i * data2.num_blocks);
    uint32_t count = 0;
    for (uint32_t b = 0; b < data2.num_blocks; ++b) {
      count += popcount_block(fp_blocks[b]);
    }
    popcounts2[i] = count;
  }

  // 3. Setup parallel processing
  unsigned int u_threads = get_num_threads(num_threads);

  size_t total_size = static_cast<size_t>(data1.num_fps) * data2.num_fps;
  std::vector<double> all_results(total_size);

  std::vector<std::thread> threads;
  size_t chunk_size = (data1.num_fps + u_threads - 1) / u_threads;

  for (unsigned int t = 0; t < u_threads; ++t) {
    size_t start_idx = t * chunk_size;
    if (start_idx >= data1.num_fps)
      continue;
    size_t end_idx =
        std::min(start_idx + chunk_size, static_cast<size_t>(data1.num_fps));

    // Calculate the pointer to this thread's result slice
    double *results_ptr = &all_results[start_idx * data2.num_fps];

    threads.emplace_back(tanimoto_mmap_worker, std::ref(data1), std::ref(data2),
                         std::ref(popcounts2), start_idx, end_idx, results_ptr);
  }

  for (auto &thread : threads) {
    thread.join();
  }

  // 4. Create and return a NumPy array with zero-copy from the C++ vector
  npy_intp dims[1] = {static_cast<npy_intp>(total_size)};
  PyObject *py_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  void *py_arr_data = PyArray_DATA(reinterpret_cast<PyArrayObject *>(py_obj));
  memcpy(py_arr_data, all_results.data(), total_size * sizeof(double));

  return py::object(py::handle<>(py_obj));
}

BOOST_PYTHON_MODULE(_smiles_fp) {

  py::def(
      "save_fingerprints", save_fingerprints_impl,
      (py::arg("py_fps"), py::arg("filename")),
      "Save a sequence of fingerprints to a binary file.\n"
      "All fingerprints must be of the same length.");

  py::def("load_fingerprints", load_fingerprints_impl,
      (py::arg("filename")),
      "Load a sequence of fingerprints from a binary file.");
  
  py::def("bulk_tanimoto_parallel", bulk_tanimoto_parallel,
      (py::arg("py_fps"),
       py::arg("py_fps2"),
       py::arg("num_threads") = -1),
      "Calculate Tanimoto similarities in parallel from RDKit fingerprint objects.");

  py::def("bulk_tanimoto_mmap", bulk_tanimoto_mmap,
      (py::arg("filename1"), py::arg("filename2"),
       py::arg("num_threads") = -1),
      "Calculate Tanimoto similarities between two binary fingerprint files.\n"
      "Uses memory-mapping.\n"
      "Faster and more memory-efficient than using the ExplicitBitVect objects directly.");
}