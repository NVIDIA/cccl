// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <nvbench_helper.cuh>

template <typename T>
static void negate_if(nvbench::state& state, nvbench::type_list<T>)
{
  const auto n       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto entropy = str_to_entropy(state.get_string("Entropy"));
  const auto val     = lerp_min_max<T>(entropy_to_probability(entropy));
  auto transform_op  = ::cuda::std::negate<T>{};
  auto select_op     = less_then_t<T>{val};

  thrust::device_vector<T> input = generate(n);
  thrust::device_vector<T> output(n, thrust::no_init);

  const auto selected_elements = thrust::count_if(input.cbegin(), input.cend(), select_op);
  state.add_element_count(n);
  state.add_global_memory_reads<T>(n);
  state.add_global_memory_writes<T>(selected_elements);

  caching_allocator_t alloc; // transform_if shouldn't allocate, but let's be consistent
  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::transform_if(policy(alloc, launch), input.begin(), input.end(), output.begin(), transform_op, select_op);
    });
}

NVBENCH_BENCH_TYPES(negate_if, NVBENCH_TYPE_AXES(integral_types))
  //  .set_name("negate_if")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
