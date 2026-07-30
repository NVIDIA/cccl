// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "nvbench_helper.cuh"

template <typename T, typename Func>
static void bench_extremum(nvbench::state& state, nvbench::type_list<T>, Func func)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in = generate(elements);

  using offset_t = typename decltype(in.cbegin())::difference_type;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<offset_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(func(policy(alloc, launch), in.cbegin(), in.cend()));
             });
}

template <typename T>
static void min_element(nvbench::state& state, nvbench::type_list<T> list)
{
  bench_extremum(state, list, [](auto&&... args) {
    return thrust::min_element(args...);
  });
}

NVBENCH_BENCH_TYPES(min_element, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("min_element")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

template <typename T>
static void max_element(nvbench::state& state, nvbench::type_list<T> list)
{
  bench_extremum(state, list, [](auto&&... args) {
    return thrust::max_element(args...);
  });
}

NVBENCH_BENCH_TYPES(max_element, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("max_element")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
