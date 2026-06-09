// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// BabelStream-style bandwidth benchmarks via cub::DeviceTransform::Transform.
// Custom ops self-register their tile substitutes via tile_eligible<>, so the
// dispatch hook routes them to the tile kernel under --enable-tile + the
// CCCL_ENABLE_TILE_TRANSFORM_DISPATCH macro.

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

// User-defined scalar ops (used at the call site, in both build modes).
struct mul_op {
    template <class B>
    __host__ __device__ auto operator()(B b) const { return -(b + b); }
};
struct add_op {
    template <class A, class B>
    __host__ __device__ auto operator()(A a, B b) const { return a + b; }
};
struct triad_op {
    template <class B, class C>
    __host__ __device__ auto operator()(B b, C c) const { return b - c - c; }
};
struct nstream_op {
    template <class A, class B, class C>
    __host__ __device__ auto operator()(A a, B b, C c) const { return a + b - c - c; }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
// Tile-friendly substitutes (must be stateless + trivially default constructible).
struct tile_mul_op {
    template <class B>
    __tile__ auto operator()(B b) const { return -(b + b); }
};
struct tile_add_op {
    template <class A, class B>
    __tile__ auto operator()(A a, B b) const { return a + b; }
};
struct tile_triad_op {
    template <class B, class C>
    __tile__ auto operator()(B b, C c) const { return b - c - c; }
};
struct tile_nstream_op {
    template <class A, class B, class C>
    __tile__ auto operator()(A a, B b, C c) const { return a + b - c - c; }
};

// Self-register each scalar op for all T (partial specialization on T).
CUB_NAMESPACE_BEGIN
namespace transform
{
template <class T> struct tile_eligible<mul_op,     T, 1> : ::cuda::std::true_type { using tile_op_type = tile_mul_op; };
template <class T> struct tile_eligible<add_op,     T, 2> : ::cuda::std::true_type { using tile_op_type = tile_add_op; };
template <class T> struct tile_eligible<triad_op,   T, 2> : ::cuda::std::true_type { using tile_op_type = tile_triad_op; };
template <class T> struct tile_eligible<nstream_op, T, 3> : ::cuda::std::true_type { using tile_op_type = tile_nstream_op; };
} // namespace transform
CUB_NAMESPACE_END
#endif

// True if `bytes_needed` worth of GPU memory is available, with 5% headroom
// for driver overhead. Caller should `state.skip(...)` on false.
inline bool gpu_mem_available(size_t bytes_needed) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    return bytes_needed + (bytes_needed / 20) < free_b;
}

template <typename T>
struct Buffers {
    T *a{}, *b{}, *c{};
    int64_t n{};
    Buffers(int64_t n) : n(n) {
        cudaMalloc(&a, n * sizeof(T));
        cudaMalloc(&b, n * sizeof(T));
        cudaMalloc(&c, n * sizeof(T));
        bench_init::rand_fill(a, n, 0xA111);
        bench_init::rand_fill(b, n, 0xB222);
        bench_init::rand_fill(c, n, 0xC333);
        cudaDeviceSynchronize();
    }
    ~Buffers() { cudaFree(a); cudaFree(b); cudaFree(c); }
};

// --- benchmarks ---
template <typename T>
void mul(nvbench::state& state, nvbench::type_list<T>) {
    auto n = state.get_int64("Elements{io}");
    Buffers<T> buf(n);
    state.add_element_count(n);
    state.add_global_memory_reads<T>(n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(buf.b), buf.c, n, mul_op{}, launch.get_stream());
    });
}

template <typename T>
void add(nvbench::state& state, nvbench::type_list<T>) {
    auto n = state.get_int64("Elements{io}");
    Buffers<T> buf(n);
    state.add_element_count(n);
    state.add_global_memory_reads<T>(2 * n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(buf.a, buf.b), buf.c, n, add_op{}, launch.get_stream());
    });
}

template <typename T>
void triad(nvbench::state& state, nvbench::type_list<T>) {
    auto n = state.get_int64("Elements{io}");
    Buffers<T> buf(n);
    state.add_element_count(n);
    state.add_global_memory_reads<T>(2 * n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(buf.b, buf.c), buf.a, n, triad_op{}, launch.get_stream());
    });
}

template <typename T>
void nstream(nvbench::state& state, nvbench::type_list<T>) {
    auto n = state.get_int64("Elements{io}");
    Buffers<T> buf(n);
    state.add_element_count(n);
    state.add_global_memory_reads<T>(3 * n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(buf.a, buf.b, buf.c), buf.a, n, nstream_op{}, launch.get_stream());
    });
}

using types = nvbench::type_list<std::int8_t, std::int16_t, float, double>;
inline auto sizes = std::vector<nvbench::int64_t>{16, 20, 24, 28, 31};

NVBENCH_BENCH_TYPES(mul,     NVBENCH_TYPE_AXES(types)).set_name("tile_mul").add_int64_power_of_two_axis("Elements{io}", sizes);
NVBENCH_BENCH_TYPES(add,     NVBENCH_TYPE_AXES(types)).set_name("tile_add").add_int64_power_of_two_axis("Elements{io}", sizes);
NVBENCH_BENCH_TYPES(triad,   NVBENCH_TYPE_AXES(types)).set_name("tile_triad").add_int64_power_of_two_axis("Elements{io}", sizes);
NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(types)).set_name("tile_nstream").add_int64_power_of_two_axis("Elements{io}", sizes);

NVBENCH_MAIN
