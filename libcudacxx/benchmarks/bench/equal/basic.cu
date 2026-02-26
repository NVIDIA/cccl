//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <cuda/memory_pool>
#include <cuda/std/__pstl_algorithm>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void range_iter(nvbench::state& state, nvbench::type_list<T>)
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

  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      do_not_optimize(
        cuda::std::equal(cuda_policy(alloc, launch), dinput.begin(), dinput.end(), cuda::constant_iterator<T>{0}));
    });
}

NVBENCH_BENCH_TYPES(range_iter, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base_range_iter")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.01});

template <typename T>
static void range_range(nvbench::state& state, nvbench::type_list<T>)
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
               do_not_optimize(cuda::std::equal(
                 cuda_policy(alloc, launch),
                 dinput.begin(),
                 dinput.end(),
                 cuda::constant_iterator<T>{0},
                 cuda::constant_iterator<T>{0, elements}));
             });
}

NVBENCH_BENCH_TYPES(range_range, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base_range_range")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.01});
