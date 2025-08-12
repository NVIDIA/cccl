// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/device_vector.h>
#include <thrust/swap.h>

#include <nvbench_helper.cuh>

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a = generate(elements);
  thrust::device_vector<T> b = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements);
  state.add_global_memory_writes<T>(2 * elements);

  caching_allocator_t alloc; // swap_ranges shouldn't allocate, but let's be consistent
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::swap_ranges(policy(alloc, launch), a.begin(), a.end(), b.begin());
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(integral_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
