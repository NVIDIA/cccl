// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

#include "common.h"

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
