// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include <omp.h>

namespace py = pybind11;

// Define a custom hash function for std::pair<int, int>
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const& pair) const {
        auto h1 = std::hash<T1>{}(pair.first);
        auto h2 = std::hash<T2>{}(pair.second);
        return h1 ^ h2;
    }
};

std::tuple<py::array_t<int>, py::array_t<int>, int> compute_indices(int n_bas, py::array_t<int> ao_loc_array, std::unordered_set<std::pair<int, int>, pair_hash> considered_indices) {
    py::buffer_info ao_loc_info = ao_loc_array.request();
    int* ao_loc_ptr = static_cast<int*>(ao_loc_info.ptr);

    py::ssize_t n_upper_bound = (py::ssize_t)(n_bas * (n_bas) / 2) * (n_bas * (n_bas - 1) / 2);
    // py::print(n_upper_bound);
    auto input_ijkl_array = py::array_t<int>({n_upper_bound, (py::ssize_t)4});
    input_ijkl_array[py::make_tuple(py::ellipsis())] = -1;
    auto output_sizes_array = py::array_t<int>({n_upper_bound, (py::ssize_t)5});
    output_sizes_array[py::make_tuple(py::ellipsis())] = -1;

    py::buffer_info input_ijkl_info = input_ijkl_array.request();
    int* input_ijkl_ptr = static_cast<int*>(input_ijkl_info.ptr);

    py::buffer_info output_sizes_info = output_sizes_array.request();
    int* output_sizes_ptr = static_cast<int*>(output_sizes_info.ptr);

    py::ssize_t num_calls = 0;
    #pragma omp parallel for reduction(+:num_calls)  // Parallelize the outermost loop
    for (py::ssize_t i = 0; i < n_bas; ++i) {
        py::ssize_t partial_num_calls = 0;
        // py::print(".", py::arg("end")="", py::arg("flush") = true); // do not print from threads
        for (py::ssize_t j = 0; j <= i; ++j) {
            for (py::ssize_t k = i; k < n_bas; ++k) {
                for (py::ssize_t l = 0; l <= k; ++l) {
                    int di = ao_loc_ptr[i + 1] - ao_loc_ptr[i];
                    int dj = ao_loc_ptr[j + 1] - ao_loc_ptr[j];
                    int dk = ao_loc_ptr[k + 1] - ao_loc_ptr[k];
                    int dl = ao_loc_ptr[l + 1] - ao_loc_ptr[l];

                    bool found_nonzero = false;
                    for (int bi = ao_loc_ptr[i]; bi < ao_loc_ptr[i + 1]; ++bi) {
                        for (int bj = ao_loc_ptr[j]; bj < ao_loc_ptr[j + 1]; ++bj) {
                            if (considered_indices.count({bi, bj}) > 0) {
                                for (int bk = ao_loc_ptr[k]; bk < ao_loc_ptr[k + 1]; ++bk) {
                                    if (bk >= bi) {
                                        int mla = ao_loc_ptr[l];
                                        if (bk == bi) {
                                            mla = std::max(bj, ao_loc_ptr[l]);
                                        }
                                        for (int bl = mla; bl < ao_loc_ptr[l + 1]; ++bl) {
                                            if (considered_indices.count({bk, bl}) > 0) {
                                                found_nonzero = true;
                                                break;
                                            }
                                        }
                                        if (found_nonzero) break;
                                    }
                                }
                                if (found_nonzero) break;
                            }
                            if (found_nonzero) break;
                        }
                        if (found_nonzero) break;
                    }
                    
                    if (!found_nonzero) continue;

                    py::ssize_t offset_ijkl = (i * (n_bas) / 2) * (n_bas * (n_bas - 1) / 2) * 4 + partial_num_calls * 4;
                    input_ijkl_ptr[offset_ijkl + 0] = i;
                    input_ijkl_ptr[offset_ijkl + 1] = j;
                    input_ijkl_ptr[offset_ijkl + 2] = k;
                    input_ijkl_ptr[offset_ijkl + 3] = l;

                    py::ssize_t offset_sizes = (i * (n_bas) / 2) * (n_bas * (n_bas - 1) / 2) * 5 +partial_num_calls * 5;
                    output_sizes_ptr[offset_sizes + 0] = di;
                    output_sizes_ptr[offset_sizes + 1] = dj;
                    output_sizes_ptr[offset_sizes + 2] = dk;
                    output_sizes_ptr[offset_sizes + 3] = dl;
                    output_sizes_ptr[offset_sizes + 4] = di * dj * dk * dl;

                    partial_num_calls += 1;
                    
                }
            }
        }
        num_calls += partial_num_calls;
    }

    return std::make_tuple(input_ijkl_array, output_sizes_array, num_calls);
}

PYBIND11_MODULE(cpp_shell_gen, m) {
    m.def("compute_indices", &compute_indices);
}