// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// PyTorch ops on tile.  Uses ct::tanh / ct::sin / ct::exp / ct::select.

#include <nvbench/nvbench.cuh>
#include "device_transform.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/tuple>
#include <vector>

#include "bench_init.cuh"

namespace ct = cuda::tiles;

// --- Unary --- (compute in float, cast back so the same ops work for __half/__bf16/float)
template <class T> __tile__ auto as_float(T v) { return ct::element_cast<float>(v); }
template <class T, class F> __tile__ auto from_float(F f) { return ct::element_cast<ct::tile_element_t<T>>(f); }

struct relu_op    { template <class T> __tile__ auto operator()(T v) const {
    auto f = as_float(v); return from_float<T>(ct::select(f > 0.0f, f, f - f)); } };
struct sigmoid_op { template <class T> __tile__ auto operator()(T v) const {
    auto f = as_float(v); return from_float<T>(1.0f / (1.0f + ct::exp(-f))); } };
struct tanh_op    { template <class T> __tile__ auto operator()(T v) const {
    return from_float<T>(ct::tanh(as_float(v))); } };
struct gelu_op    { template <class T> __tile__ auto operator()(T v) const {
    constexpr float k0 = 0.7978845608028654f, k1 = 0.044715f;
    auto f = as_float(v);
    return from_float<T>(0.5f * f * (1.0f + ct::tanh(k0 * (f + k1 * f * f * f)))); } };
struct sin_op { template <class T> __tile__ auto operator()(T v) const { return from_float<T>(ct::sin(as_float(v))); } };
struct exp_op { template <class T> __tile__ auto operator()(T v) const { return from_float<T>(ct::exp(as_float(v))); } };

// --- Binary ---
struct binary_add  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a + b; } };
struct binary_sub  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a - b; } };
struct binary_mul  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a * b; } };
struct binary_div  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a / b; } };
// le/ge: cast the bool result tile to A's element type so it fits the float output buffer
//        (CUB does the same implicit cast via its iterator path).
struct binary_le   { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::element_cast<ct::tile_element_t<A>>(a <= b); } };
struct binary_ge   { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::element_cast<ct::tile_element_t<A>>(a >= b); } };
struct binary_fmin { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::select(a < b, a, b); } };
struct binary_fmax { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::select(a > b, a, b); } };


template <typename Op, typename T, bool MufuHeavy = false>
void run_unary(nvbench::state& state) {
    const auto n = state.get_int64("Elements{io}");
    T *in, *out;
    cudaMalloc(&in, n * sizeof(T)); cudaMalloc(&out, n * sizeof(T));
    bench_init::rand_fill(in, n, 0xA111); cudaDeviceSynchronize();
    state.add_element_count(n);
    state.add_global_memory_reads<T>(n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub_tile::DeviceTransform::Transform<0, MufuHeavy>(
            ::cuda::std::make_tuple(in), out, n, Op{}, launch.get_stream());
    });
    cudaFree(in); cudaFree(out);
}

template <typename Op, typename T>
void run_binary(nvbench::state& state) {
    const auto n = state.get_int64("Elements{io}");
    T *a, *b, *out;
    cudaMalloc(&a, n*sizeof(T)); cudaMalloc(&b, n*sizeof(T)); cudaMalloc(&out, n*sizeof(T));
    bench_init::rand_fill(a, n, 0xA111);
    bench_init::rand_fill(b, n, 0xB222);
    cudaDeviceSynchronize();
    state.add_element_count(n);
    state.add_global_memory_reads<T>(2*n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub_tile::DeviceTransform::Transform(
            ::cuda::std::make_tuple(a, b), out, n, Op{}, launch.get_stream());
    });
    cudaFree(a); cudaFree(b); cudaFree(out);
}

using element_types = nvbench::type_list<__half, __nv_bfloat16, float>;
inline auto pt_sizes = std::vector<nvbench::int64_t>{16, 20, 24, 28, 31};

#define UNARY_BENCH(name, op, mufu) \
    template <typename T> void name##_bench(nvbench::state& state, nvbench::type_list<T>) { run_unary<op, T, mufu>(state); } \
    NVBENCH_BENCH_TYPES(name##_bench, NVBENCH_TYPE_AXES(element_types)).set_name("tile_" #name).add_int64_power_of_two_axis("Elements{io}", pt_sizes);

// MufuHeavy hint set for ops dominated by MUFU intrinsics (exp/tanh/sin/cos).
// relu is just compare+select, so no hint.
UNARY_BENCH(relu,    relu_op,    false)
UNARY_BENCH(sigmoid, sigmoid_op, true)
UNARY_BENCH(tanh,    tanh_op,    true)
UNARY_BENCH(gelu,    gelu_op,    true)
UNARY_BENCH(sin,     sin_op,     true)
UNARY_BENCH(exp,     exp_op,     true)

#define BINARY_BENCH(name, op) \
    template <typename T> void name##_bench(nvbench::state& state, nvbench::type_list<T>) { run_binary<op, T>(state); } \
    NVBENCH_BENCH_TYPES(name##_bench, NVBENCH_TYPE_AXES(element_types)).set_name("tile_pt_" #name).add_int64_power_of_two_axis("Elements{io}", pt_sizes);

BINARY_BENCH(add,  binary_add)
BINARY_BENCH(sub,  binary_sub)
BINARY_BENCH(mul,  binary_mul)
BINARY_BENCH(div,  binary_div)
BINARY_BENCH(le,   binary_le)
BINARY_BENCH(ge,   binary_ge)
BINARY_BENCH(fmin, binary_fmin)
BINARY_BENCH(fmax, binary_fmax)

NVBENCH_MAIN
