// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Grayscale: RGB pixel -> luminance via three separate input streams.
// Custom rgb_to_y op self-registers its tile substitute via tile_eligible<>.

#include <nvbench/nvbench.cuh>

#include <cub/device/device_transform.cuh>

#include <cuda_runtime.h>
#include <cuda/std/tuple>
#include <vector>

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

#include "bench_init.cuh"

struct rgb_to_y {
    template <class R, class G, class B>
    __host__ __device__ auto operator()(R r, G g, B b) const {
        constexpr float w_r = 0.2989f;
        constexpr float w_g = 0.587f;
        constexpr float w_b = 0.114f;
        return w_r * r + w_g * g + w_b * b;
    }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
struct tile_rgb_to_y {
    template <class R, class G, class B>
    __tile__ auto operator()(R r, G g, B b) const {
        constexpr float w_r = 0.2989f;
        constexpr float w_g = 0.587f;
        constexpr float w_b = 0.114f;
        return w_r * r + w_g * g + w_b * b;
    }
};

CUB_NAMESPACE_BEGIN
namespace transform
{
template <class T> struct tile_eligible<rgb_to_y, T, 3> : ::cuda::std::true_type {};
template <> struct tile_operator<rgb_to_y> { using type = tile_rgb_to_y; };
} // namespace transform
CUB_NAMESPACE_END
#endif

template <typename T>
void grayscale(nvbench::state& state, nvbench::type_list<T>) {
    const auto n = state.get_int64("Elements{io}");
    T *r, *g, *b, *out;
    cudaMalloc(&r, n*sizeof(T)); cudaMalloc(&g, n*sizeof(T)); cudaMalloc(&b, n*sizeof(T));
    cudaMalloc(&out, n*sizeof(T));
    bench_init::rand_fill(r, n, 0xA111);
    bench_init::rand_fill(g, n, 0xA222);
    bench_init::rand_fill(b, n, 0xA333);

    state.add_element_count(n);
    state.add_global_memory_reads<T>(3 * n);   // matches CUB's rgb_t<T> = 3*sizeof(T)
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(r, g, b), out, n, rgb_to_y{}, launch.get_stream());
    });
    cudaFree(r); cudaFree(g); cudaFree(b); cudaFree(out);
}

using value_types = nvbench::type_list<float, double>;

NVBENCH_BENCH_TYPES(grayscale, NVBENCH_TYPE_AXES(value_types)).set_name("tile_grayscale")
    .add_int64_power_of_two_axis("Elements{io}", std::vector<nvbench::int64_t>{16, 20, 24, 28, 31});

NVBENCH_MAIN
