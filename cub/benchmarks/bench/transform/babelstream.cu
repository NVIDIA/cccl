// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_copy.cuh>

// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_ALGORITHM alg 0:4:1

#include <cub/device/dispatch/dispatch_transform.cuh>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  if CUB_DETAIL_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "This benchmark does not support being compiled for multiple architectures"
#  endif

#  if TUNE_ALGORITHM == 4 && __CUDA_ARCH_LIST__ < 900
#    error "Cannot compile algorithm 4 (ublkcp) below sm90"
#  endif

#  if TUNE_ALGORITHM != 2 && TUNE_ITEMS_PER_THREAD != 1
#    error "Skip ITEMS_PER_THREAD tuning for algorithms other than 2"
#  endif

struct policy_hub_t
{
  struct max_policy : cub::ChainedPolicy<350, max_policy, max_policy>
  {
    static constexpr int min_bif    = cub::detail::transform::arch_to_min_bif(__CUDA_ARCH_LIST__);
    static constexpr auto algorithm = static_cast<cub::detail::transform::Algorithm>(TUNE_ALGORITHM);
    using algo_policy               = ::cuda::std::_If<
                    Alg == Algorithm::fallback_for,
                    void,
                    ::cuda::std::_If<Alg == Algorithm::prefetch,
                                     cub::detail::transform::prefetch_policy_t<TUNE_THREADS>,
                                     ::cuda::std::_If<Alg == Algorithm::unrolled_staged,
                                                      cub::detail::transform::unrolled_policy_t<TUNE_THREADS, TUNE_ITEMS_PER_THREAD>,
                                                      cub::detail::transform::async_copy_policy_t<TUNE_THREADS>>>>;
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
auto array_size_powers = nvbench::range(16, 28, 4);

template <typename OffsetT, typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
CUB_RUNTIME_FUNCTION static void bench_transform(
  nvbench::state& state,
  OffsetT count,
  ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
  RandomAccessIteratorOut output,
  TransformOp transform_op)
{
#if !TUNE_BASE
  using policy_t = policy_hub_t;
#else
  using policy_t = cub::detail::transform::policy_hub<RandomAccessIteratorsIn...>;
#endif

  state.exec(nvbench::exec_tag::no_batch, [&](const nvbench::launch& launch) {
    cub::detail::transform::dispatch_t<false,
                                       OffsetT,
                                       ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                       RandomAccessIteratorOut,
                                       TransformOp,
                                       policy_t>::dispatch(count, inputs, output, transform_op, launch.get_stream());
  });
}

// Modified from BabelStream to also work for integers
constexpr auto startA      = 1; // BabelStream: 0.1
constexpr auto startB      = 2; // BabelStream: 0.2
constexpr auto startC      = 3; // BabelStream: 0.1
constexpr auto startScalar = 4; // BabelStream: 0.4

template <typename T>
static void mul(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);

  const T scalar = startScalar;
  bench_transform(state, n, ::cuda::std::tuple{c.begin()}, b.begin(), [=] __device__ __host__(const T& ci) {
    return ci * scalar;
  });
}

NVBENCH_BENCH_TYPES(mul, NVBENCH_TYPE_AXES(element_types))
  .set_name("mul")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T>
static void add(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  bench_transform(
    state, n, ::cuda::std::tuple{a.begin(), b.begin()}, c.begin(), [] _CCCL_DEVICE(const T& ai, const T& bi) -> T {
      return ai + bi;
    });
}

NVBENCH_BENCH_TYPES(add, NVBENCH_TYPE_AXES(element_types))
  .set_name("add")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T>
static void triad(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);
  const T scalar = startScalar;
  bench_transform(
    state, n, ::cuda::std::tuple{b.begin(), c.begin()}, a.begin(), [=] _CCCL_DEVICE(const T& bi, const T& ci) {
      return bi + scalar * ci;
    });
}

NVBENCH_BENCH_TYPES(triad, NVBENCH_TYPE_AXES(element_types))
  .set_name("triad")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

template <typename T>
static void nstream(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);
  const T scalar = startScalar;
  bench_transform(
    state,
    n,
    ::cuda::std::tuple{a.begin(), b.begin(), c.begin()},
    a.begin(),
    [=] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) {
      return ai + bi + scalar * ci;
    });
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);

// TODO(bgruber): I guess we should split this into a seperate file? But it should be tuned together with the rest here.
static void heavy(nvbench::state& state)
{
  using T                  = uint32_t; // must be unsigned so overflow is defined
  constexpr auto heavyness = 128;

  const auto n = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in(n, startA);
  thrust::device_vector<T> out(n, startB);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(n);
  bench_transform(state, n, ::cuda::std::tuple{in.begin()}, out.begin(), [=] _CCCL_DEVICE(T data) {
    T reg[heavyness];
    reg[0] = data;
    for (int i = 1; i < heavyness; ++i)
    {
      reg[i] = reg[i - 1] * reg[i - 1] + 1;
    }
    for (int i = 0; i < heavyness; ++i)
    {
      reg[i] = (reg[i] * reg[i]) % 19;
    }
    for (int i = 0; i < heavyness; ++i)
    {
      reg[i] = reg[heavyness - i - 1] * reg[i];
    }
    T x = 0;
    for (int i = 0; i < heavyness; ++i)
    {
      x += reg[i];
    }
    return x;
  });
}

NVBENCH_BENCH(heavy).set_name("heavy").add_int64_power_of_two_axis("Elements{io}", array_size_powers);
