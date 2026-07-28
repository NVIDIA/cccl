// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! `cub::DeviceFind::UpperBoundSortedValues`: haystack and values (needles) sorted.

#include <cub/device/device_find.cuh>

#include <cstdint>

#include <nvbench_helper.cuh>

#include "find_bound_common.cuh"

template <typename T>
void basic(nvbench::state& state, nvbench::type_list<T>)
{
  bounds_bench_data<T> s(state);
  s.sort_needles();

  state.add_element_count(s.needles);
  state.add_global_memory_reads<T>(s.elements + s.needles);
  state.add_global_memory_writes<std::ptrdiff_t>(s.needles);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_CUDA_API(
      cub::DeviceFind::UpperBoundSortedValues,
      "UpperBoundSortedValues failed",
      s.range_ptr(),
      static_cast<std::int64_t>(s.elements),
      s.values_ptr(),
      static_cast<std::int64_t>(s.needles),
      s.output_ptr(),
      less_t{},
      env);
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(integral_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_axis("NeedlesRatio", {1, 25, 50});
