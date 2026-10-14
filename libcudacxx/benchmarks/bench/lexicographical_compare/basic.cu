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

// Identical zero-filled inputs except for one bumped element at `violation_point`,
// which makes that index the first non-equivalent pair under operator<.
// The early-terminating implementation should read at most 2 * violation_point
// elements (one from each range) before returning.
template <typename T>
static void prepare_inputs(thrust::device_vector<T>& d1, thrust::device_vector<T>& d2, std::size_t violation_point)
{
  thrust::fill(d1.begin(), d1.end(), T{0});
  thrust::fill(d2.begin(), d2.end(), T{0});
  if (violation_point < d2.size())
  {
    d2[violation_point] = T{1};
  }
}

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto violation_frac  = state.get_float64("ViolationAt");
  const auto violation_point = cuda::std::clamp<std::size_t>(elements * violation_frac, 0ull, elements - 1);

  thrust::device_vector<T> dinput1(elements, thrust::no_init);
  thrust::device_vector<T> dinput2(elements, thrust::no_init);
  prepare_inputs(dinput1, dinput2, violation_point);

  state.add_global_memory_reads<T>(2 * violation_point);
  state.add_global_memory_writes<size_t>(1);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(cuda::std::lexicographical_compare(
                 cuda_policy(alloc, launch), dinput1.begin(), dinput1.end(), dinput2.begin(), dinput2.end()));
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

  thrust::device_vector<T> dinput1(elements, thrust::no_init);
  thrust::device_vector<T> dinput2(elements, thrust::no_init);
  prepare_inputs(dinput1, dinput2, violation_point);

  state.add_global_memory_reads<T>(2 * violation_point);
  state.add_global_memory_writes<size_t>(1);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    do_not_optimize(cuda::std::lexicographical_compare(
      cuda_policy(alloc, launch), dinput1.begin(), dinput1.end(), dinput2.begin(), dinput2.end(), cuda::std::less<>{}));
  });
}

NVBENCH_BENCH_TYPES(with_predicate, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("with_predicate")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("ViolationAt", std::vector{1.0, 0.5, 0.01});
