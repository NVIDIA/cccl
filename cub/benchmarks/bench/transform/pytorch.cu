// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// %RANGE% TUNE_ALGORITHM alg 0:4:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

// those parameters only apply if TUNE_ALGORITHM == 1 (vectorized)
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1
// %RANGE% TUNE_VECTORS_PER_THREAD vpt 1:4:1

#if !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)
#  error "Non-vectorized algorithms require vector size and vectors per thread to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1 || TUNE_VECTORS_PER_THREAD != 1)

#include "common.h"

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
  float>;
#endif

template <typename Op, typename T>
static void unary(nvbench::state& state, nvbench::type_list<T>)
try
{
  using OffsetT = int64_t;

  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> in(n, 1337);
  thrust::device_vector<T> out(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, ::cuda::std::tuple{in.begin()}, out.begin(), static_cast<OffsetT>(n), Op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

#define BENCHMARK_UNARY(func)                                               \
  template <typename T>                                                     \
  static void func##_bench(nvbench::state& state, nvbench::type_list<T> tl) \
  {                                                                         \
    unary<func##_op>(state, tl);                                            \
  }                                                                         \
                                                                            \
  NVBENCH_BENCH_TYPES(func##_bench, NVBENCH_TYPE_AXES(element_types))       \
    .set_name(#func)                                                        \
    .set_type_axes_names({"T{ct}"})                                         \
    .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

struct relu_op
{
  template <typename T>
  _CCCL_API auto operator()(T value) const
  {
    return ::cuda::std::max(value, T{0});
  }
};
BENCHMARK_UNARY(relu);

struct sigmoid_op
{
  template <typename T>
  _CCCL_API auto operator()(T value) const
  {
    return T{1} / (T{1} + ::cuda::std::exp(-value));
  }
};
BENCHMARK_UNARY(sigmoid);

struct tanh_op
{
  template <typename T>
  _CCCL_API auto operator()(T value) const
  {
    return ::cuda::std::tanh(value);
  }
};
BENCHMARK_UNARY(tanh);

struct gelu_op
{
  template <typename T>
  _CCCL_API auto operator()(T value) const
  {
    return value * T{0.5} * (T{1} + ::cuda::std::erf(value * T{M_SQRT1_2}));
  }
};
BENCHMARK_UNARY(gelu);

struct sin_op
{
  template <typename T>
  _CCCL_API auto operator()(T value) const
  {
    return ::cuda::std::sin(value);
  }
};
BENCHMARK_UNARY(sin);

struct exp_op
{
  template <typename T>
  _CCCL_API auto operator()(T value) const
  {
    return ::cuda::std::exp(value);
  }
};
BENCHMARK_UNARY(exp);

template <typename Op, typename T>
static void binary(nvbench::state& state, nvbench::type_list<T>)
try
{
  using OffsetT = int64_t;

  const auto n = state.get_int64("Elements{io}");
  thrust::device_vector<T> in1(n, 1337);
  thrust::device_vector<T> in2(n, 42);
  thrust::device_vector<T> out(n, thrust::no_init);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, ::cuda::std::tuple{in1.begin(), in2.begin()}, out.begin(), static_cast<OffsetT>(n), Op{});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

#define BENCHMARK_BINARY(func)                                              \
  template <typename T>                                                     \
  static void func##_bench(nvbench::state& state, nvbench::type_list<T> tl) \
  {                                                                         \
    binary<func##_op>(state, tl);                                           \
  }                                                                         \
                                                                            \
  NVBENCH_BENCH_TYPES(func##_bench, NVBENCH_TYPE_AXES(element_types))       \
    .set_name(#func)                                                        \
    .set_type_axes_names({"T{ct}"})                                         \
    .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

using add_op = cuda::std::plus<>;
BENCHMARK_BINARY(add);

using sub_op = cuda::std::minus<>;
BENCHMARK_BINARY(sub);

using mul_op = cuda::std::multiplies<>;
BENCHMARK_BINARY(mul);

using div_op = cuda::std::divides<>;
BENCHMARK_BINARY(div);

using le_op = cuda::std::less_equal<>;
BENCHMARK_BINARY(le);

using ge_op = cuda::std::greater_equal<>;
BENCHMARK_BINARY(ge);

struct fmin_op
{
  template <typename T>
  _CCCL_API auto operator()(T a, T b) const
  {
    return ::cuda::std::fmin(a, b);
  }
};
BENCHMARK_BINARY(fmin);

struct fmax_op
{
  template <typename T>
  _CCCL_API auto operator()(T a, T b) const
  {
    return ::cuda::std::fmax(a, b);
  }
};
BENCHMARK_BINARY(fmax);
