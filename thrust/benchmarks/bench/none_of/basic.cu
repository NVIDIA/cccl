// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/logical.h>

#include <cuda/memory_pool>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <class T>
struct equal_to_val
{
  T val_;

  constexpr equal_to_val(const T& val) noexcept
      : val_(val)
  {}

  __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return val == val_;
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
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

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(thrust::none_of(policy(alloc, launch), dinput.begin(), dinput.end(), equal_to_val{val}));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.01});
