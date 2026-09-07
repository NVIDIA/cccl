// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
try
{
  // A 32-bit offset type or the value 0 or 0xFF... have <1% performance impact
  const auto value     = T{42};
  const auto n         = state.get_int64("Elements{io}");
  const bool unaligned = state.get_string("Aligned") == "no";
  thrust::device_vector<T> out(n + unaligned);

  state.add_element_count(n);
  state.add_global_memory_reads<T>(0);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, cuda::std::tuple{}, out.begin() + unaligned, n, return_constant<T>{value});
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(fill, NVBENCH_TYPE_AXES(integral_types))
  .set_name("fill")
  .set_type_axes_names({"T{ct}"})
  .add_string_axis("Aligned", {"yes", "no"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 32, 4));
