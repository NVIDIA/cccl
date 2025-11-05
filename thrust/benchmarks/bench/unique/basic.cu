// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  const std::size_t min_segment_size = 1;
  const std::size_t max_segment_size = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<T> input = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);
  thrust::device_vector<T> output(elements);

  caching_allocator_t alloc;
  // not a warm-up run, we need to run once to determine the size of the output
  const auto new_end             = thrust::unique_copy(policy(alloc), input.cbegin(), input.cend(), output.begin());
  const std::size_t unique_items = ::cuda::std::distance(output.begin(), new_end);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(unique_items);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::unique_copy(policy(alloc, launch), input.cbegin(), input.cend(), output.begin());
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
