// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  auto generator               = generate(elements);
  thrust::device_vector<T> lhs = generator;
  thrust::device_vector<T> rhs = generator;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements * 2);
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::inner_product(policy(alloc, launch), lhs.begin(), lhs.end(), rhs.begin(), T{0});
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(all_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
