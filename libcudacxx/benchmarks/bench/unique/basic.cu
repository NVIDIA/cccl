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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <cuda/memory_pool>
#include <cuda/std/execution>
#include <cuda/stream_ref>

#include "nvbench_helper.cuh"

// Input with runs of equal elements: 0,0,1,1,2,2,... (segment size 2)
template <typename T>
static void make_unique_input(thrust::device_vector<T>& in, std::size_t elements)
{
  in.resize(elements);
  thrust::transform(
    thrust::counting_iterator<std::size_t>(0),
    thrust::counting_iterator<std::size_t>(elements),
    in.begin(),
    [] __device__(std::size_t i) {
      return static_cast<T>(i / 2);
    });
}

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in;
  make_unique_input(in, elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  // unique writes at most elements
  state.add_global_memory_writes<T>(elements / 2);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               do_not_optimize(cuda::std::unique(cuda_policy(alloc, launch), in.begin(), in.end()));
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

template <typename T>
static void with_comp(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in;
  make_unique_input(in, elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  // unique writes at most elements
  state.add_global_memory_writes<T>(elements / 2);

  caching_allocator_t alloc{};

  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      do_not_optimize(cuda::std::unique(cuda_policy(alloc, launch), in.begin(), in.end(), cuda::std::equal_to<T>{}));
    });
}

NVBENCH_BENCH_TYPES(with_comp, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("with_comp")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
