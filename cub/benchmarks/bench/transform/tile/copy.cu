// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Pure copy bench (identity transform). Custom identity op self-registers
// its tile substitute via tile_eligible<>; under --enable-tile + the
// dispatch macro this routes to the tile load_masked/store_masked path,
// otherwise it falls through to CUB's standard transform.

#include <nvbench/nvbench.cuh>

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>

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
template <class T> struct tile_eligible<identity, T, 1> : ::cuda::std::true_type {};
template <> struct tile_operator<identity> { using type = tile_identity; };
} // namespace transform
CUB_NAMESPACE_END
#endif

template <typename T>
void copy(nvbench::state& state, nvbench::type_list<T>) {
    const auto n = state.get_int64("Elements{io}");
    thrust::device_vector<T> in(n), out(n);
    T* in_ptr  = thrust::raw_pointer_cast(in.data());
    T* out_ptr = thrust::raw_pointer_cast(out.data());
    bench_init::rand_fill(in_ptr, n, 0xA111); cudaDeviceSynchronize();
    state.add_element_count(n);
    state.add_global_memory_reads<T>(n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(in_ptr), out_ptr, n, identity{}, launch.get_stream());
    });
}

using types = nvbench::type_list<std::int8_t, std::int16_t, std::int32_t, float, double>;
inline auto sizes = std::vector<nvbench::int64_t>{16, 20, 24, 28, 31};

NVBENCH_BENCH_TYPES(copy, NVBENCH_TYPE_AXES(types))
    .set_name("tile_copy")
    .add_int64_power_of_two_axis("Elements{io}", sizes);

NVBENCH_MAIN
