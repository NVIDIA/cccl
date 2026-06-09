// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// PyTorch-style ops via cub::DeviceTransform::Transform. Each custom op
// self-registers a tile substitute through tile_eligible<>, so the dispatch
// hook routes them to the tile kernel under --enable-tile + the
// CCCL_ENABLE_TILE_TRANSFORM_DISPATCH macro. MUFU-heavy ops also opt into
// tile_mufu_heavy<> so the tile policy picker caps items/thread at the
// vector width on sub-4-byte types.

#include <nvbench/nvbench.cuh>

#include <cub/device/device_transform.cuh>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/cmath>
#include <cuda/std/tuple>
#include <vector>

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

#include "bench_init.cuh"

// ========================================================================
// Scalar ops (the types the user passes to cub::DeviceTransform::Transform).
// Sub-4-byte input types compute in float and cast back, matching the tile
// substitute below.
// ========================================================================
template <class T> __host__ __device__ float to_f(T v) { return static_cast<float>(v); }
template <class T> __host__ __device__ T from_f(float f) { return static_cast<T>(f); }

struct relu_op    { template <class T> __host__ __device__ T operator()(T v) const {
    float f = to_f(v); return from_f<T>(f > 0.0f ? f : 0.0f); } };
struct sigmoid_op { template <class T> __host__ __device__ T operator()(T v) const {
    float f = to_f(v); return from_f<T>(1.0f / (1.0f + ::cuda::std::exp(-f))); } };
struct tanh_op    { template <class T> __host__ __device__ T operator()(T v) const {
    return from_f<T>(::cuda::std::tanh(to_f(v))); } };
struct gelu_op    { template <class T> __host__ __device__ T operator()(T v) const {
    constexpr float k0 = 0.7978845608028654f, k1 = 0.044715f;
    float f = to_f(v);
    return from_f<T>(0.5f * f * (1.0f + ::cuda::std::tanh(k0 * (f + k1 * f * f * f)))); } };
struct sin_op     { template <class T> __host__ __device__ T operator()(T v) const {
    return from_f<T>(::cuda::std::sin(to_f(v))); } };
struct exp_op     { template <class T> __host__ __device__ T operator()(T v) const {
    return from_f<T>(::cuda::std::exp(to_f(v))); } };

struct binary_add  { template <class A, class B> __host__ __device__ auto operator()(A a, B b) const { return a + b; } };
struct binary_sub  { template <class A, class B> __host__ __device__ auto operator()(A a, B b) const { return a - b; } };
struct binary_mul  { template <class A, class B> __host__ __device__ auto operator()(A a, B b) const { return a * b; } };
struct binary_div  { template <class A, class B> __host__ __device__ auto operator()(A a, B b) const { return a / b; } };
struct binary_le   { template <class A, class B> __host__ __device__ A operator()(A a, B b) const { return static_cast<A>(a <= b); } };
struct binary_ge   { template <class A, class B> __host__ __device__ A operator()(A a, B b) const { return static_cast<A>(a >= b); } };
struct binary_fmin { template <class A, class B> __host__ __device__ auto operator()(A a, B b) const { return a < b ? a : b; } };
struct binary_fmax { template <class A, class B> __host__ __device__ auto operator()(A a, B b) const { return a > b ? a : b; } };

// ========================================================================
// Tile substitutes + trait registration. Only compiled under tile mode.
// ========================================================================
#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
namespace ct = ::cuda::tiles;

template <class T> __tile__ auto as_float(T v) { return ct::element_cast<float>(v); }
template <class T, class F> __tile__ auto from_float(F f) { return ct::element_cast<ct::tile_element_t<T>>(f); }

struct tile_relu    { template <class T> __tile__ auto operator()(T v) const {
    auto f = as_float(v); return from_float<T>(ct::select(f > 0.0f, f, f - f)); } };
struct tile_sigmoid { template <class T> __tile__ auto operator()(T v) const {
    auto f = as_float(v); return from_float<T>(1.0f / (1.0f + ct::exp(-f))); } };
struct tile_tanh    { template <class T> __tile__ auto operator()(T v) const {
    return from_float<T>(ct::tanh(as_float(v))); } };
struct tile_gelu    { template <class T> __tile__ auto operator()(T v) const {
    constexpr float k0 = 0.7978845608028654f, k1 = 0.044715f;
    auto f = as_float(v);
    return from_float<T>(0.5f * f * (1.0f + ct::tanh(k0 * (f + k1 * f * f * f)))); } };
struct tile_sin     { template <class T> __tile__ auto operator()(T v) const { return from_float<T>(ct::sin(as_float(v))); } };
struct tile_exp     { template <class T> __tile__ auto operator()(T v) const { return from_float<T>(ct::exp(as_float(v))); } };

struct tile_binary_add  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a + b; } };
struct tile_binary_sub  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a - b; } };
struct tile_binary_mul  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a * b; } };
struct tile_binary_div  { template <class A, class B> __tile__ auto operator()(A a, B b) const { return a / b; } };
struct tile_binary_le   { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::element_cast<ct::tile_element_t<A>>(a <= b); } };
struct tile_binary_ge   { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::element_cast<ct::tile_element_t<A>>(a >= b); } };
struct tile_binary_fmin { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::select(a < b, a, b); } };
struct tile_binary_fmax { template <class A, class B> __tile__ auto operator()(A a, B b) const { return ct::select(a > b, a, b); } };

CUB_NAMESPACE_BEGIN
namespace transform
{
// Unary
template <class T> struct tile_eligible<relu_op,    T, 1> : ::cuda::std::true_type { using tile_op_type = tile_relu;    };
template <class T> struct tile_eligible<sigmoid_op, T, 1> : ::cuda::std::true_type { using tile_op_type = tile_sigmoid; };
template <class T> struct tile_eligible<tanh_op,    T, 1> : ::cuda::std::true_type { using tile_op_type = tile_tanh;    };
template <class T> struct tile_eligible<gelu_op,    T, 1> : ::cuda::std::true_type { using tile_op_type = tile_gelu;    };
template <class T> struct tile_eligible<sin_op,     T, 1> : ::cuda::std::true_type { using tile_op_type = tile_sin;     };
template <class T> struct tile_eligible<exp_op,     T, 1> : ::cuda::std::true_type { using tile_op_type = tile_exp;     };

// MUFU-heavy unary ops: hint to tile policy picker to cap items/thread at vector width on sub-4-byte types.
template <> struct tile_mufu_heavy<sigmoid_op> : ::cuda::std::true_type {};
template <> struct tile_mufu_heavy<tanh_op>    : ::cuda::std::true_type {};
template <> struct tile_mufu_heavy<gelu_op>    : ::cuda::std::true_type {};
template <> struct tile_mufu_heavy<sin_op>     : ::cuda::std::true_type {};
template <> struct tile_mufu_heavy<exp_op>     : ::cuda::std::true_type {};

// Binary
template <class T> struct tile_eligible<binary_add,  T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_add;  };
template <class T> struct tile_eligible<binary_sub,  T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_sub;  };
template <class T> struct tile_eligible<binary_mul,  T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_mul;  };
template <class T> struct tile_eligible<binary_div,  T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_div;  };
template <class T> struct tile_eligible<binary_le,   T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_le;   };
template <class T> struct tile_eligible<binary_ge,   T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_ge;   };
template <class T> struct tile_eligible<binary_fmin, T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_fmin; };
template <class T> struct tile_eligible<binary_fmax, T, 2> : ::cuda::std::true_type { using tile_op_type = tile_binary_fmax; };
} // namespace transform
CUB_NAMESPACE_END
#endif

// ========================================================================
// Bench harness.
// ========================================================================
template <typename Op, typename T>
void run_unary(nvbench::state& state) {
    const auto n = state.get_int64("Elements{io}");
    T *in, *out;
    cudaMalloc(&in, n * sizeof(T)); cudaMalloc(&out, n * sizeof(T));
    bench_init::rand_fill(in, n, 0xA111); cudaDeviceSynchronize();
    state.add_element_count(n);
    state.add_global_memory_reads<T>(n);
    state.add_global_memory_writes<T>(n);
    state.exec([&](nvbench::launch& launch) {
        cub::DeviceTransform::Transform(
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
        cub::DeviceTransform::Transform(
            ::cuda::std::make_tuple(a, b), out, n, Op{}, launch.get_stream());
    });
    cudaFree(a); cudaFree(b); cudaFree(out);
}

using element_types = nvbench::type_list<__half, __nv_bfloat16, float>;
inline auto pt_sizes = std::vector<nvbench::int64_t>{16, 20, 24, 28, 31};

#define UNARY_BENCH(name, op) \
    template <typename T> void name##_bench(nvbench::state& state, nvbench::type_list<T>) { run_unary<op, T>(state); } \
    NVBENCH_BENCH_TYPES(name##_bench, NVBENCH_TYPE_AXES(element_types)).set_name("tile_" #name).add_int64_power_of_two_axis("Elements{io}", pt_sizes);

UNARY_BENCH(relu,    relu_op)
UNARY_BENCH(sigmoid, sigmoid_op)
UNARY_BENCH(tanh,    tanh_op)
UNARY_BENCH(gelu,    gelu_op)
UNARY_BENCH(sin,     sin_op)
UNARY_BENCH(exp,     exp_op)

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
