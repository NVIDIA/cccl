// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <cuda/memory_pool>
#include <cuda/stream>

#include "nvbench_helper.cuh"

template <class T>
struct plus_one
{
  template <class U>
  [[nodiscard]] __device__ constexpr T operator()(const U val) const noexcept
  {
    return static_cast<T>(val + 1);
  }
};

template <typename T>
static void unary(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in = generate(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(1);

  caching_allocator_t alloc{};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    do_not_optimize(
      thrust::transform_reduce(policy(alloc, launch), in.begin(), in.end(), plus_one<T>{}, 42, cuda::std::plus<T>{}));
  });
}

NVBENCH_BENCH_TYPES(unary, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
