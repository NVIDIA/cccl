// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<T> input = generate(elements, entropy);

  thrust::device_vector<T> vec(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               vec = input;
               timer.start();
               thrust::sort(policy(alloc, launch), vec.begin(), vec.end(), less_t{});
               timer.stop();
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
