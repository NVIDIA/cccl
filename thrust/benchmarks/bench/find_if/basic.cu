// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/detail/internal_functional.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/find.h>

#include "nvbench_helper.cuh"

template <typename T>
void find_if(nvbench::state& state, nvbench::type_list<T>)
{
  T val = 1;
  // set up input
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = static_cast<std::size_t>(elements * common_prefix);

  thrust::device_vector<T> dinput(elements, thrust::no_init);
  thrust::fill(dinput.begin(), dinput.begin() + mismatch_point, T{0});
  thrust::fill(dinput.begin() + mismatch_point, dinput.end(), val);

  state.add_global_memory_reads<T>(mismatch_point + 1);
  state.add_global_memory_writes<size_t>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    thrust::find_if(policy(alloc, launch), dinput.begin(), dinput.end(), thrust::detail::equal_to_value<T>(val));
  });
}

NVBENCH_BENCH_TYPES(find_if, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.0});
