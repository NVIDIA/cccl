//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = cuda::std::clamp<std::size_t>(elements * common_prefix, 0, elements - 2);

  thrust::device_vector<T> in(elements, thrust::no_init);
  thrust::sequence(in.begin(), in.end(), 0);
  in[mismatch_point] = in[mismatch_point + 1];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(mismatch_point);
  state.add_global_memory_writes<T>(0);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(cuda::std::adjacent_find(cuda_policy(alloc, launch), in.cbegin(), in.cend()));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.01});

template <typename T>
static void with_comp(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = cuda::std::clamp<std::size_t>(elements * common_prefix, 0, elements - 2);

  thrust::device_vector<T> in(elements, thrust::no_init);
  thrust::sequence(in.begin(), in.end(), 0);
  in[mismatch_point] = in[mismatch_point + 1];

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(mismatch_point);
  state.add_global_memory_writes<T>(0);

  caching_allocator_t alloc;
  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      do_not_optimize(
        cuda::std::adjacent_find(cuda_policy(alloc, launch), in.cbegin(), in.cend(), ::cuda::std::greater<T>{}));
    });
}

NVBENCH_BENCH_TYPES(with_comp, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("with_comp")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.01});
