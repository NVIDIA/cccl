//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of the localized_array class which is used to allocate a piece
 * of data that is dispatched over multiple data places
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/__stf/utility/memory.cuh>
#include <cuda/experimental/__stf/utility/traits.cuh>

#include <list>
#include <random>

namespace cuda::experimental::stf
{

struct current_place_desc {
    current_place_desc() = default;
    current_place_desc(pos4 place_position, dim4 grid_dims) : place_position(mv(place_position)), grid_dims(mv(grid_dims)) {}

    // Position of the exec place in the grid
    pos4 place_position;
    // Dimension of the grid of execution places
    dim4 grid_dims;
};

namespace reserved {

// An array in device memory used to store the information about execution
// place per CUDA kernel. This is a fixed size array, and we are going to use
// its entries in a round-robin fashion, and use the %%gridid assembly variable
// to retrieve the index of the CUDA kernel
__device__ current_place_desc _per_kernel_place_desc[1024];

inline __device__ unsigned int get_grid_id() {
    unsigned int grid_id;
    asm("mov.u32 %0, %%gridid;" : "=r"(grid_id));
    return grid_id;
}

inline __device__ void save_current_place(pos4 place_position, dim4 grid_dims) {
    auto &desc = _per_kernel_place_desc[get_grid_id() % 1024];
    desc.place_position = mv(place_position);
    desc.grid_dims = mv(grid_dims);
}

inline __device__ pos4 get_current_place() {
    return _per_kernel_place_desc[get_grid_id() % 1024].place_position;
}

inline __device__ dim4 get_grid_dims() {
    return _per_kernel_place_desc[get_grid_id() % 1024].grid_dims;
}

}  // end namespace reserved

class comm_matrix_tracer {
public:
    comm_matrix_tracer() = default;

    void init(int _nexec_places, int _ndata_places) {
        nexec_places = _nexec_places;
        ndata_places = _ndata_places;

        cuda_safe_call(cudaMallocManaged((void**) &base, nexec_places * ndata_places * sizeof(int)));
        memset(base, 0, nexec_places * ndata_places * sizeof(int));
    }

    void init(int nplaces) {
        init(nplaces, nplaces);
    }

    template <typename partitioner_t, typename shape_t>
    __device__ void mark_access(
            pos4 data_coords, const shape_t& data_shape, partitioner_t /* anonymous */, int value = 1) const {
        pos4 dev = reserved::get_current_place();
        dim4 grid_dims = reserved::get_grid_dims();
        dim4 data_dims = data_shape.get_data_dims();
        pos4 X_where = partitioner_t::get_executor(data_coords, data_dims, grid_dims);

        atomicAdd(&base[dev.x * ndata_places + X_where.x], value);
    }

    void dump(::std::string filename = "comm_matrix.dat") {
        FILE* f = fopen(filename.c_str(), "w+");
        assert(f);
        for (size_t i = 0; i < ndata_places; i++)
            for (size_t j = 0; j < nexec_places; j++) {
                if (base[j * ndata_places + i] > 0)
                    fprintf(f, "%ld,%ld,%d\n", i, j, base[j * ndata_places + i]);
            }
        fclose(f);
    }

private:
    mutable int* base;  // a pointer in managed memory
    int nexec_places;
    int ndata_places;
};

} // end namespace cuda::experimental::stf
