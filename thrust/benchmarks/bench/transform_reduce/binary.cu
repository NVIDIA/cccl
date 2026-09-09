// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "nvbench_helper.cuh"

// Benchmarks for the binary (two-input) transform_reduce overload.
// This computes: reduce(init, transform(*first1, *first2), transform(*(first1+1), *(first2+1)), ...)

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> lhs = generate(elements);
  thrust::device_vector<T> rhs = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements * 2);
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc;
  // Use the 6-argument overload with explicit operators (inner product-style: plus + multiplies)
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(thrust::transform_reduce(
                 policy(alloc, launch),
                 lhs.begin(),
                 lhs.end(),
                 rhs.begin(),
                 T{},
                 ::cuda::std::plus<T>{},
                 ::cuda::std::multiplies<T>{}));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

template <typename T>
static void custom_ops(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> lhs = generate(elements);
  thrust::device_vector<T> rhs = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements * 2);
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc;
  // Custom operators: sum of element-wise additions
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    do_not_optimize(thrust::transform_reduce(
      policy(alloc, launch), lhs.begin(), lhs.end(), rhs.begin(), T{}, ::cuda::std::plus<T>{}, ::cuda::std::plus<T>{}));
  });
}

NVBENCH_BENCH_TYPES(custom_ops, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("custom_ops")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
