// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/count.h>
#include <thrust/fill.h>

#include "nvbench_helper.cuh"

template <typename T>
struct equals
{
  T val;

  __device__ __host__ bool operator()(T i)
  {
    return i == val;
  }
};

template <typename T>
void count_if(nvbench::state& state, nvbench::type_list<T>)
{
  T val = 1;
  // set up input
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = elements * common_prefix;

  thrust::device_vector<T> dinput(elements, 0);
  thrust::fill(dinput.begin() + mismatch_point, dinput.end(), val);
  ///

  caching_allocator_t alloc;
  thrust::count_if(policy(alloc), dinput.begin(), dinput.end(), equals<T>{val});

  state.exec(nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    thrust::count_if(policy(alloc, launch), dinput.begin(), dinput.end(), equals<T>{val});
  });
}

NVBENCH_BENCH_TYPES(count_if, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("thrust::count_if")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.0});
