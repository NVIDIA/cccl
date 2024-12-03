// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:1:1

#include "common.h"

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
