// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "nvbench_helper.cuh"

template <typename T>
struct custom_op
{
  T val;

  custom_op() = delete;

  explicit custom_op(T val)
      : val(val)
  {}

  __device__ T operator()(const T& lhs, const T& rhs)
  {
    return lhs * rhs + val; // Hope to gen mad
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::adjacent_difference(
                 policy(alloc, launch), input.cbegin(), input.cend(), output.begin(), custom_op<T>{42});
             });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
