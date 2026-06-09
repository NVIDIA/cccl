// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Pure copy bench (identity transform). Custom identity op self-registers
// its tile substitute via tile_eligible<>; under --enable-tile + the
// dispatch macro this routes to the tile load_masked/store_masked path,
// otherwise it falls through to CUB's standard transform.

#include <nvbench/nvbench.cuh>

#include <cub/device/device_transform.cuh>

#include <cuda_runtime.h>
#include <cuda/std/tuple>
#include <vector>
#include <cstdint>

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

#include "bench_init.cuh"

struct identity {
    template <class T> __host__ __device__ auto operator()(T v) const { return v; }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
struct tile_identity {
    template <class T> __tile__ auto operator()(T v) const { return v; }
};

CUB_NAMESPACE_BEGIN
namespace transform
{
template <class T> struct tile_eligible<identity, T, 1> : ::cuda::std::true_type { using tile_op_type = tile_identity; };
} // namespace transform
CUB_NAMESPACE_END
#endif

template <typename T>
void copy(nvbench::state& state, nvbench::type_list<T>) {
    auto n = state.get_int64("Elements{io}");
    T *in, *out;
    cudaMalloc(&in, n * sizeof(T)); cudaMalloc(&out, n * sizeof(T));
    bench_init::rand_fill(in, n, 0xA111); cudaDeviceSynchronize();
    state.add_element_count(n);
    state.add_global_memory_reads<T>(n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(in), out, n, identity{}, launch.get_stream());
    });
    cudaFree(in); cudaFree(out);
}

using types = nvbench::type_list<std::int8_t, std::int16_t, std::int32_t, float, double>;
inline auto sizes = std::vector<nvbench::int64_t>{16, 20, 24, 28, 31};

NVBENCH_BENCH_TYPES(copy, NVBENCH_TYPE_AXES(types))
    .set_name("tile_copy")
    .add_int64_power_of_two_axis("Elements{io}", sizes);

NVBENCH_MAIN
