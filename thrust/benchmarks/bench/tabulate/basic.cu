// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>

#include <nvbench_helper.cuh>

#include "thrust/detail/raw_pointer_cast.h"

template <typename T>
static void sequence(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> output(elements, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               // sequence is implemented via thrust::tabulate
               thrust::sequence(policy(alloc, launch), output.begin(), output.end());
             });
}

NVBENCH_BENCH_TYPES(sequence, NVBENCH_TYPE_AXES(integral_types))
  .set_name("sequence")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));

template <class T>
struct seg_size_t
{
  T* d_offsets{};

  template <class OffsetT>
  __device__ T operator()(OffsetT i)
  {
    return static_cast<T>(d_offsets[i + 1] - d_offsets[i]);
  }
};

template <typename T>
static void seg_size(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<T> input(elements + 1);
  thrust::device_vector<T> output(elements, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements + 1);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  seg_size_t<T> op{thrust::raw_pointer_cast(input.data())};
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::tabulate(policy(alloc, launch), output.begin(), output.end(), op);
             });
}

NVBENCH_BENCH_TYPES(seg_size, NVBENCH_TYPE_AXES(integral_types))
  .set_name("seg_size")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
