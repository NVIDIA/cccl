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
#include <thrust/swap.h>

#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in1 = generate(elements);
  thrust::device_vector<T> in2 = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements);
  state.add_global_memory_writes<T>(2 * elements);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cuda::std::swap_ranges(cuda_policy(alloc, launch), in1.begin(), in1.end(), in2.begin());
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

template <typename T>
static void with_iter_swap(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in1 = generate(elements);
  thrust::device_vector<T> in2 = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(2 * elements);
  state.add_global_memory_writes<T>(2 * elements);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cuda::std::swap_ranges(
                 cuda_policy(alloc, launch),
                 cuda::std::reverse_iterator{in1.end()},
                 cuda::std::reverse_iterator{in1.begin()},
                 cuda::std::reverse_iterator{in2.end()});
             });
}

NVBENCH_BENCH_TYPES(with_iter_swap, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("with_iter_swap")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
