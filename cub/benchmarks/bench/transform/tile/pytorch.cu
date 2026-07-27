// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile variant of the PyTorch-style transform benches. Each named op registers a tile_operator
// substitute (gated); MUFU-heavy ops also opt into tile_mufu_heavy_v so the tile policy picker caps
// items/thread at the vector width on sub-4-byte types. Under --enable-tile +
// CCCL_ENABLE_EXPERIMENTAL_TILE_TRANSFORM_DISPATCH the dispatch hook routes them to the tile kernel; otherwise this
// is the standard CUB path. This file disappears once tile dispatch is fully transparent.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/std/cmath>

#include "../common.h"

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
#  include <cuda_tile.h>
#endif

// Scalar ops the user passes to Transform. Sub-4-byte input types compute in float and cast back,
// matching the tile substitutes below.
template <class T>
__host__ __device__ float to_f(T v)
{
  return static_cast<float>(v);
}
template <class T>
__host__ __device__ T from_f(float f)
{
  return static_cast<T>(f);
}

struct relu_op
{
  template <class T>
  __host__ __device__ T operator()(T v) const
  {
    float f = to_f(v);
    return from_f<T>(f > 0.0f ? f : 0.0f);
  }
};
struct sigmoid_op
{
  template <class T>
  __host__ __device__ T operator()(T v) const
  {
    float f = to_f(v);
    return from_f<T>(1.0f / (1.0f + ::cuda::std::exp(-f)));
  }
};
struct tanh_op
{
  template <class T>
  __host__ __device__ T operator()(T v) const
  {
    return from_f<T>(::cuda::std::tanh(to_f(v)));
  }
};
struct gelu_op
{
  template <class T>
  __host__ __device__ T operator()(T v) const
  {
    constexpr float k0 = 0.7978845608028654f, k1 = 0.044715f;
    float f = to_f(v);
    return from_f<T>(0.5f * f * (1.0f + ::cuda::std::tanh(k0 * (f + k1 * f * f * f))));
  }
};
struct sin_op
{
  template <class T>
  __host__ __device__ T operator()(T v) const
  {
    return from_f<T>(::cuda::std::sin(to_f(v)));
  }
};
struct exp_op
{
  template <class T>
  __host__ __device__ T operator()(T v) const
  {
    return from_f<T>(::cuda::std::exp(to_f(v)));
  }
};

struct binary_add
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a + b;
  }
};
struct binary_sub
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a - b;
  }
};
struct binary_mul
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a * b;
  }
};
struct binary_div
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a / b;
  }
};
struct binary_le
{
  template <class A, class B>
  __host__ __device__ A operator()(A a, B b) const
  {
    return static_cast<A>(a <= b);
  }
};
struct binary_ge
{
  template <class A, class B>
  __host__ __device__ A operator()(A a, B b) const
  {
    return static_cast<A>(a >= b);
  }
};
struct binary_fmin
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a < b ? a : b;
  }
};
struct binary_fmax
{
  template <class A, class B>
  __host__ __device__ auto operator()(A a, B b) const
  {
    return a > b ? a : b;
  }
};

#if _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()
namespace ct = ::cuda::tiles;

template <class T>
__tile__ auto as_float(T v)
{
  return ct::element_cast<float>(v);
}
template <class T, class F>
__tile__ auto from_float(F f)
{
  return ct::element_cast<ct::tile_element_t<T>>(f);
}

struct tile_relu
{
  template <class T>
  __tile__ auto operator()(T v) const
  {
    auto f = as_float(v);
    return from_float<T>(ct::select(f > 0.0f, f, f - f));
  }
};
struct tile_sigmoid
{
  template <class T>
  __tile__ auto operator()(T v) const
  {
    auto f = as_float(v);
    return from_float<T>(1.0f / (1.0f + ct::exp(-f)));
  }
};
struct tile_tanh
{
  template <class T>
  __tile__ auto operator()(T v) const
  {
    return from_float<T>(ct::tanh(as_float(v)));
  }
};
struct tile_gelu
{
  template <class T>
  __tile__ auto operator()(T v) const
  {
    constexpr float k0 = 0.7978845608028654f, k1 = 0.044715f;
    auto f = as_float(v);
    return from_float<T>(0.5f * f * (1.0f + ct::tanh(k0 * (f + k1 * f * f * f))));
  }
};
struct tile_sin
{
  template <class T>
  __tile__ auto operator()(T v) const
  {
    return from_float<T>(ct::sin(as_float(v)));
  }
};
struct tile_exp
{
  template <class T>
  __tile__ auto operator()(T v) const
  {
    return from_float<T>(ct::exp(as_float(v)));
  }
};

struct tile_binary_add
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a + b;
  }
};
struct tile_binary_sub
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a - b;
  }
};
struct tile_binary_mul
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a * b;
  }
};
struct tile_binary_div
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a / b;
  }
};
struct tile_binary_le
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return ct::element_cast<ct::tile_element_t<A>>(a <= b);
  }
};
struct tile_binary_ge
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return ct::element_cast<ct::tile_element_t<A>>(a >= b);
  }
};
struct tile_binary_fmin
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return ct::select(a < b, a, b);
  }
};
struct tile_binary_fmax
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return ct::select(a > b, a, b);
  }
};

CUB_NAMESPACE_BEGIN
namespace detail::transform::tile
{
// Unary
template <class T>
inline constexpr bool tile_eligible_v<relu_op, T, 1> = true;
template <class T>
inline constexpr bool tile_eligible_v<sigmoid_op, T, 1> = true;
template <class T>
inline constexpr bool tile_eligible_v<tanh_op, T, 1> = true;
template <class T>
inline constexpr bool tile_eligible_v<gelu_op, T, 1> = true;
template <class T>
inline constexpr bool tile_eligible_v<sin_op, T, 1> = true;
template <class T>
inline constexpr bool tile_eligible_v<exp_op, T, 1> = true;
template <>
struct tile_operator<relu_op>
{
  using type = tile_relu;
};
template <>
struct tile_operator<sigmoid_op>
{
  using type = tile_sigmoid;
};
template <>
struct tile_operator<tanh_op>
{
  using type = tile_tanh;
};
template <>
struct tile_operator<gelu_op>
{
  using type = tile_gelu;
};
template <>
struct tile_operator<sin_op>
{
  using type = tile_sin;
};
template <>
struct tile_operator<exp_op>
{
  using type = tile_exp;
};

// MUFU-heavy unary ops: hint the tile policy picker to cap items/thread at the vector width on
// sub-4-byte types.
template <>
inline constexpr bool tile_mufu_heavy_v<sigmoid_op> = true;
template <>
inline constexpr bool tile_mufu_heavy_v<tanh_op> = true;
template <>
inline constexpr bool tile_mufu_heavy_v<gelu_op> = true;
template <>
inline constexpr bool tile_mufu_heavy_v<sin_op> = true;
template <>
inline constexpr bool tile_mufu_heavy_v<exp_op> = true;

// Binary
template <class T>
inline constexpr bool tile_eligible_v<binary_add, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_sub, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_mul, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_div, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_le, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_ge, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_fmin, T, 2> = true;
template <class T>
inline constexpr bool tile_eligible_v<binary_fmax, T, 2> = true;
template <>
struct tile_operator<binary_add>
{
  using type = tile_binary_add;
};
template <>
struct tile_operator<binary_sub>
{
  using type = tile_binary_sub;
};
template <>
struct tile_operator<binary_mul>
{
  using type = tile_binary_mul;
};
template <>
struct tile_operator<binary_div>
{
  using type = tile_binary_div;
};
template <>
struct tile_operator<binary_le>
{
  using type = tile_binary_le;
};
template <>
struct tile_operator<binary_ge>
{
  using type = tile_binary_ge;
};
template <>
struct tile_operator<binary_fmin>
{
  using type = tile_binary_fmin;
};
template <>
struct tile_operator<binary_fmax>
{
  using type = tile_binary_fmax;
};
} // namespace detail::transform::tile
CUB_NAMESPACE_END
#endif // _CCCL_CUB_TILE_TRANSFORM_DISPATCH_ENABLED()

#ifdef TUNE_T
using element_types = nvbench::type_list<TUNE_T>;
#else
using element_types = nvbench::type_list<
#  if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
  __half,
#  endif
#  if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
  __nv_bfloat16,
#  endif
  nvbench::float32_t>;
#endif

template <typename Op, typename T>
static void run_unary(nvbench::state& state)
try
{
  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> in(n, T(1));
  thrust::device_vector<T> out(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{in.begin()}, out.begin(), n, Op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

template <typename Op, typename T>
static void run_binary(nvbench::state& state)
try
{
  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> a(n, T(1));
  thrust::device_vector<T> b(n, T(1));
  thrust::device_vector<T> out(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, cuda::std::tuple{a.begin(), b.begin()}, out.begin(), n, Op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

inline auto pt_sizes = nvbench::range(16, 32, 4);

#define UNARY_BENCH(name, op)                                            \
  template <typename T>                                                  \
  static void name##_bench(nvbench::state& state, nvbench::type_list<T>) \
  {                                                                      \
    run_unary<op, T>(state);                                             \
  }                                                                      \
  NVBENCH_BENCH_TYPES(name##_bench, NVBENCH_TYPE_AXES(element_types))    \
    .set_name("tile_" #name)                                             \
    .set_type_axes_names({"T{ct}"})                                      \
    .add_int64_power_of_two_axis("Elements{io}", pt_sizes)

UNARY_BENCH(relu, relu_op);
UNARY_BENCH(sigmoid, sigmoid_op);
UNARY_BENCH(tanh, tanh_op);
UNARY_BENCH(gelu, gelu_op);
UNARY_BENCH(sin, sin_op);
UNARY_BENCH(exp, exp_op);

#define BINARY_BENCH(name, op)                                           \
  template <typename T>                                                  \
  static void name##_bench(nvbench::state& state, nvbench::type_list<T>) \
  {                                                                      \
    run_binary<op, T>(state);                                            \
  }                                                                      \
  NVBENCH_BENCH_TYPES(name##_bench, NVBENCH_TYPE_AXES(element_types))    \
    .set_name("tile_pt_" #name)                                          \
    .set_type_axes_names({"T{ct}"})                                      \
    .add_int64_power_of_two_axis("Elements{io}", pt_sizes)

BINARY_BENCH(add, binary_add);
BINARY_BENCH(sub, binary_sub);
BINARY_BENCH(mul, binary_mul);
BINARY_BENCH(div, binary_div);
BINARY_BENCH(le, binary_le);
BINARY_BENCH(ge, binary_ge);
BINARY_BENCH(fmin, binary_fmin);
BINARY_BENCH(fmax, binary_fmax);
