// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <nvbench_helper.cuh>

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  using select_op_t = less_then_t<T>;

  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  const T val = lerp_min_max<T>(entropy_to_probability(entropy));
  select_op_t select_op{val};

  thrust::device_vector<T> input = generate(elements);
  const auto selected_elements   = thrust::count_if(input.cbegin(), input.cend(), select_op);
  thrust::device_vector<T> output(selected_elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(selected_elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::copy_if(policy(alloc, launch), input.cbegin(), input.cend(), output.begin(), select_op);
             });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.000"});
