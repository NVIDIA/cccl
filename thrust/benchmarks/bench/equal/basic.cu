// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in(elements, T{1});
  thrust::device_vector<T> in_equal = in;
  thrust::device_vector<T> in_different(elements, T{2});

  caching_allocator_t alloc;

  thrust::device_vector<T> in_halfway_equal(elements, T{1});
  thrust::fill(policy(alloc), in_halfway_equal.begin() + elements / 2, in_halfway_equal.end(), T{2});

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements);

  auto body = [&](auto&... args) {
    do_not_optimize(thrust::equal(policy(alloc, args...), in.begin(), in.end(), in_equal.begin()));
    do_not_optimize(thrust::equal(policy(alloc, args...), in.begin(), in.end(), in_different.begin()));
    do_not_optimize(thrust::equal(policy(alloc, args...), in.begin(), in.end(), in_halfway_equal.begin()));
  };
  body(); // discarded warmup run

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    body(launch);
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
