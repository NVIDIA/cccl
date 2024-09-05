// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:2:1

#include "common.h"

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

// Modified from BabelStream to also work for integers and to make nstream maintain a consistent workload since it
// overwrites one input array. If the data changed at each iteration, the performance would be unstable.
inline constexpr auto startA      = 11; // BabelStream: 0.1
inline constexpr auto startB      = 2; // BabelStream: 0.2
inline constexpr auto startC      = 1; // BabelStream: 0.1
inline constexpr auto startScalar = -2; // BabelStream: 0.4

static_assert(startA == (startA + startB + startScalar * startC), "nstream must have a consistent workload");

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
  const auto n = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a(n, startA);
  thrust::device_vector<T> b(n, startB);
  thrust::device_vector<T> c(n, startC);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(3 * n);
  state.add_global_memory_writes<T>(n);
  const T scalar = startScalar;
  bench_transform(
    state,
    ::cuda::std::tuple{a.begin(), b.begin(), c.begin()},
    a.begin(),
    n,
    [=] _CCCL_DEVICE(const T& ai, const T& bi, const T& ci) {
      return ai + bi + scalar * ci;
    });
}

NVBENCH_BENCH_TYPES(nstream, NVBENCH_TYPE_AXES(element_types, offset_types))
  .set_name("nstream")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", array_size_powers);
