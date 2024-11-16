// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "nvbench_helper.cuh"

template <typename T>
static void benchmark(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));
  thrust::device_vector<T> a(elements, T{1});
  thrust::device_vector<T> b(elements, T{1});

  const auto common_prefix = state.get_float64("CommonPrefixRatio");
  const auto same_elements = std::min(static_cast<std::size_t>(elements * common_prefix), elements);
  caching_allocator_t alloc;
  thrust::fill(policy(alloc), b.begin() + same_elements, b.end(), T{2});

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * std::max(same_elements, std::size_t(1))); // using `same_elements` instead
                                                                                 // of `elements` corresponds to the
                                                                                 // actual elements read in an early
                                                                                 // exit
  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    do_not_optimize(thrust::equal(policy(alloc, launch), a.begin(), a.end(), b.begin()));
  });
}

NVBENCH_BENCH_TYPES(benchmark, NVBENCH_TYPE_AXES(integral_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("CommonPrefixRatio", std::vector{1.0, 0.5, 0.0});
