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

#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto midpoint_float = state.get_float64("ShiftedTo");
  const auto midpoint       = static_cast<std::size_t>(elements * midpoint_float);

  thrust::device_vector<T> in = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements - midpoint);
  state.add_global_memory_writes<T>(elements - midpoint);

  caching_allocator_t alloc{};
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(cuda::std::shift_right(cuda_policy(alloc, launch), in.begin(), in.end(), midpoint));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("ShiftedTo", std::vector{0.9, 0.6, 0.45, 0.01});
