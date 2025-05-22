// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

// keep checks at the top so compilation of discarded variants fails really fast
#include <cub/device/dispatch/dispatch_transform.cuh>
#if !TUNE_BASE && TUNE_ALGORITHM == 1
#  if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "When tuning, this benchmark does not support being compiled for multiple architectures"
#  endif
#  if (__CUDA_ARCH_LIST__) < 900
#    error "Cannot compile algorithm 4 (ublkcp) below sm90"
#  endif
#  ifndef _CUB_HAS_TRANSFORM_UBLKCP
#    error "Cannot tune for ublkcp algorithm, which is not provided by CUB (old CTK?)"
#  endif
#endif

#include <cuda/std/type_traits>

#include <stdexcept>

#include <nvbench_helper.cuh>

template <typename... RandomAccessIteratorsIn>
#if TUNE_BASE
using policy_hub_t = cub::detail::transform::policy_hub<false, ::cuda::std::tuple<RandomAccessIteratorsIn...>>;
#else
struct policy_hub_t
{
  struct max_policy : cub::ChainedPolicy<500, max_policy, max_policy>
  {
    static constexpr int min_bif    = cub::detail::transform::arch_to_min_bytes_in_flight(__CUDA_ARCH_LIST__);
    static constexpr auto algorithm = static_cast<cub::detail::transform::Algorithm>(TUNE_ALGORITHM);
    using algo_policy =
      ::cuda::std::_If<algorithm == cub::detail::transform::Algorithm::prefetch,
                       cub::detail::transform::prefetch_policy_t<TUNE_THREADS>,
                       cub::detail::transform::async_copy_policy_t<TUNE_THREADS, __CUDA_ARCH_LIST__ == 900 ? 128 : 16>>;
  };
};
#endif

#ifdef TUNE_T
using element_types = nvbench::type_list<TUNE_T>;
#else
using element_types =
  nvbench::type_list<std::int8_t,
                     std::int16_t,
                     float,
                     double
#  ifdef NVBENCH_HELPER_HAS_I128
                     ,
                     __int128
#  endif
                     >;
#endif

// BabelStream uses 2^25, H200 can fit 2^31 int128s
// 2^20 chars / 2^16 int128 saturate V100 (min_bif =12 * SM count =80)
// 2^21 chars / 2^17 int128 saturate A100 (min_bif =16 * SM count =108)
// 2^23 chars / 2^19 int128 saturate H100/H200 HBM3 (min_bif =32or48 * SM count =132)
// inline auto array_size_powers = std::vector<nvbench::int64_t>{28};
inline auto array_size_powers = nvbench::range(16, 28, 4);

template <typename OffsetT,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename ExecTag = decltype(nvbench::exec_tag::no_batch)>
void bench_transform(
  nvbench::state& state,
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  OffsetT num_items,
  TransformOp transform_op,
  ExecTag exec_tag = nvbench::exec_tag::no_batch)
{
  state.exec(nvbench::exec_tag::gpu | exec_tag, [&](const nvbench::launch& launch) {
    cub::detail::transform::dispatch_t<
      cub::detail::transform::requires_stable_address::no,
      OffsetT,
      ::cuda::std::tuple<RandomAccessIteratorsIn...>,
      RandomAccessIteratorOut,
      TransformOp,
      policy_hub_t<RandomAccessIteratorsIn...>>::dispatch(inputs, output, num_items, transform_op, launch.get_stream());
  });
}

// Modified from BabelStream to also work for integers
inline constexpr auto startA      = 1; // BabelStream: 0.1
inline constexpr auto startB      = 2; // BabelStream: 0.2
inline constexpr auto startC      = 3; // BabelStream: 0.1
inline constexpr auto startScalar = 4; // BabelStream: 0.4

// TODO(bgruber): we should put those somewhere into libcu++:
// from C++ GSL
struct narrowing_error : std::runtime_error
{
  narrowing_error()
      : std::runtime_error("Narrowing error")
  {}
};

// from C++ GSL
// implementation inspired by: https://github.com/microsoft/GSL/blob/main/include/gsl/narrow
template <typename DstT, typename SrcT, ::cuda::std::enable_if_t<::cuda::std::is_arithmetic_v<SrcT>, int> = 0>
constexpr DstT narrow(SrcT value)
{
  constexpr bool is_different_signedness = ::cuda::std::is_signed_v<SrcT> != ::cuda::std::is_signed_v<DstT>;
  const auto converted                   = static_cast<DstT>(value);
  if (static_cast<SrcT>(converted) != value || (is_different_signedness && ((converted < DstT{}) != (value < SrcT{}))))
  {
    throw narrowing_error{};
  }
  return converted;
}

template <typename T, typename OffsetT>
static void mul(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(state, ::cuda::std::tuple{c.begin()}, b.begin(), n, [=] _CCCL_DEVICE(const T& ci) {
    return ci * scalar;
  });
}

NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("mul")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T, typename OffsetT>
static void add(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  bench_transform(
    state, ::cuda::std::tuple{a.begin(), b.begin()}, c.begin(), n, [] _CCCL_DEVICE(const T& ai, const T& bi) -> T {
      return ai + bi;
    });
}

NVBENCH_BENCH_TYPES(add, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("add")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T, typename OffsetT>
static void triad(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  const T scalar = startScalar;
  bench_transform(
    state, ::cuda::std::tuple{b.begin(), c.begin()}, a.begin(), n, [=] _CCCL_DEVICE(const T& bi, const T& ci) {
      return bi + scalar * ci;
    });
}

NVBENCH_BENCH_TYPES(triad, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("triad")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T, typename OffsetT>
static void nstream(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n         = narrow<OffsetT>(state.get_int64("Elements{io}"));
  const auto overwrite = static_cast<bool>(state.get_int64("OverwriteInput"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  // The BabelStream nstream overwrites one input array to avoid write-allocation of cache lines. However, this changes
  // the data that is computed for each iteration and results in an unstable workload. Therefore, we added an axis to
  // choose a different output array. Pass `-a OverwriteInput=0` to the benchmark to disable overwriting the input.
  thrust::device_vector<T> d;
  if (!overwrite)
  {
    d.resize(n);
  }

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);
  const T scalar = startScalar;
  bench_transform(
    state,
    ::cuda::std::tuple{a.begin(), b.begin(), c.begin()},
    overwrite ? a.begin() : d.begin(),
    n,
    [=] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) {
      return ai + bi + scalar * ci;
    },
    nvbench::exec_tag::none); // Use batch mode for benchmarking since the workload changes. Not necessary when
                              // OverwriteInput=0, but doesn't hurt
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers)
  .add_int64_axis("OverwriteInput", {1});

// This benchmark tests overlapping memory regions for reading and is compute intensive

template <typename OffsetT>
static void compare_complex(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  const auto n                      = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<complex> in = generate(n);
  thrust::device_vector<bool> out(n - 1);

  state.add_element_count(n);
  state.add_global_memory_reads<complex>(n);
  state.add_global_memory_writes<bool>(n);

  // the complex comparison needs lots of compute and transform reads from overlapping input
  using compare_op = less_t;
  bench_transform(state, ::cuda::std::tuple{in.begin(), in.begin() + 1}, out.begin(), n - 1, compare_op{});
}

// TODO(bgruber): hardcode OffsetT?
NVBENCH_BENCH_TYPES(compare_complex, NVBENCH_TYPE_AXES(offset_types))
  .set_name("compare_complex")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

// This benchmark is compute intensive with diverging threads

template <class IndexT, class OutputT>
struct fib_t
{
  __device__ OutputT operator()(IndexT n)
  {
    OutputT t1 = 0;
    OutputT t2 = 1;

    if (n < 1)
    {
      return t1;
    }
    if (n == 1)
    {
      return t1;
    }
    if (n == 2)
    {
      return t2;
    }
    for (IndexT i = 3; i <= n; ++i)
    {
      const auto next = t1 + t2;
      t1              = t2;
      t2              = next;
    }
    return t2;
  }
};

template <typename OffsetT>
static void fibonacci(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  using index_t                     = int64_t;
  using output_t                    = uint32_t;
  const auto n                      = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<index_t> in = generate(n, bit_entropy::_1_000, index_t{0}, index_t{42});
  thrust::device_vector<output_t> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<index_t>(n);
  state.add_global_memory_writes<output_t>(n);

  bench_transform(state, ::cuda::std::tuple{in.begin()}, out.begin(), n, fib_t<index_t, output_t>{});
}

NVBENCH_BENCH_TYPES(fibonacci, NVBENCH_TYPE_AXES(offset_types))
  .set_name("fibonacci")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

// This benchmark uses a LOT of registers and is compute intensive.

template <int N>
struct heavy_functor
{
  // we need to use an unsigned type so overflow in arithmetic wraps around
  __device__ std::uint32_t operator()(std::uint32_t data) const
  {
    std::uint32_t reg[N];
    reg[0] = data;
    for (int i = 1; i < N; ++i)
    {
      reg[i] = reg[i - 1] * reg[i - 1] + 1;
    }
    for (int i = 0; i < N; ++i)
    {
      reg[i] = (reg[i] * reg[i]) % 19;
    }
    for (int i = 0; i < N; ++i)
    {
      reg[i] = reg[N - i - 1] * reg[i];
    }
    std::uint32_t x = 0;
    for (int i = 0; i < N; ++i)
    {
      x += reg[i];
    }
    return x;
  }
};

template <typename Heaviness>
static void heavy(nvbench::state& state, nvbench::type_list<Heaviness>)
{
  using value_t                     = std::uint32_t;
  using offset_t                    = int;
  const auto n                      = narrow<offset_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<value_t> in = generate(n);
  thrust::device_vector<value_t> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<value_t>(n);
  state.add_global_memory_writes<value_t>(n);

  bench_transform(state, ::cuda::std::tuple{in.begin()}, out.begin(), n, heavy_functor<Heaviness::value>{});
}

using ::cuda::std::integral_constant;
#ifdef TUNE_Heaviness
using heaviness = nvbench::type_list<TUNE_Heaviness>; // expands to "integral_constant<int, ...>"
#else
using heaviness =
  nvbench::type_list<integral_constant<int, 32>,
                     integral_constant<int, 64>,
                     integral_constant<int, 128>,
                     integral_constant<int, 256>>;
#endif

NVBENCH_BENCH_TYPES(heavy, NVBENCH_TYPE_AXES(heaviness))
  .set_name("heavy")
  .set_type_axes_names({"Heaviness{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
