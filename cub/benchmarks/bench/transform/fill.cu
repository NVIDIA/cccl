// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Because CUB cannot inspect the transformation function, we cannot add any tunings based on the results of this
// benchmark. Its main use is to detect regressions.

// %RANGE% TUNE_THREADS tpb 128:1024:128
// %RANGE% TUNE_ALGORITHM alg 0:3:1

#include "common.h"

template <typename T>
struct return_constant
{
  T value;

  _CCCL_DEVICE auto operator()() const -> T
  {
    return value;
  }
};

template <typename T>
static void fill(nvbench::state& state, nvbench::type_list<T>)
{
  // A 32-bit offset type or the value 0 or 0xFF... have <1% performance impact
  using offset_t   = int64_t;
  const auto value = T{42};
  const auto n     = narrow<offset_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> out(n);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(0);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, ::cuda::std::tuple{}, out.begin(), n, return_constant<T>{value});
}

NVBENCH_BENCH_TYPES(fill, NVBENCH_TYPE_AXES(integral_types))
  .set_name("fill")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
