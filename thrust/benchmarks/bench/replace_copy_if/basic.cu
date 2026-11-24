// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>
#include <thrust/replace.h>

#include <cuda/memory_pool>
#include <cuda/stream>

#include "nvbench_helper.cuh"

struct equal_to_42
{
  template <class T>
  __device__ constexpr bool operator()(const T& val) const noexcept
  {
    return val == static_cast<T>(42);
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> in = generate(elements, bit_entropy::_1_000, T{0}, T{42});
  thrust::device_vector<T> out(elements, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc{};

  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      do_not_optimize(
        thrust::replace_copy_if(policy(alloc, launch), in.begin(), in.end(), out.begin(), equal_to_42{}, 1337));
    });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
