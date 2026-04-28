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

#include <cuda/functional>
#include <cuda/std/execution>
#include <cuda/stream>

#include "nvbench_helper.cuh"

// All-zero is a valid heap; setting one element to 1 forces a violation at
// that child index since its parent is still 0.
template <typename T>
static void prepare_input(thrust::device_vector<T>& d, std::size_t violation_point)
{
  thrust::fill(d.begin(), d.end(), T{0});
  if (violation_point >= 1 && violation_point < d.size())
  {
    d[violation_point] = T{1};
  }
}

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto violation_frac  = state.get_float64("ViolationAt");
  const auto violation_point = cuda::std::clamp<std::size_t>(elements * violation_frac, 0ull, elements - 1);

  thrust::device_vector<T> dinput(elements, thrust::no_init);
  prepare_input(dinput, violation_point);

  state.add_global_memory_reads<T>(2 * violation_point);
  state.add_global_memory_writes<size_t>(1);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(cuda::std::is_heap_until(cuda_policy(alloc, launch), dinput.begin(), dinput.end()));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("ViolationAt", std::vector{1.0, 0.5, 0.01});

template <typename T>
static void with_predicate(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto violation_frac  = state.get_float64("ViolationAt");
  const auto violation_point = cuda::std::clamp<std::size_t>(elements * violation_frac, 0ull, elements - 1);

  thrust::device_vector<T> dinput(elements, thrust::no_init);
  prepare_input(dinput, violation_point);

  state.add_global_memory_reads<T>(2 * violation_point);
  state.add_global_memory_writes<size_t>(1);

  caching_allocator_t alloc{};

  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      do_not_optimize(
        cuda::std::is_heap_until(cuda_policy(alloc, launch), dinput.begin(), dinput.end(), cuda::std::less<>{}));
    });
}

NVBENCH_BENCH_TYPES(with_predicate, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("with_predicate")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("ViolationAt", std::vector{1.0, 0.5, 0.01});
