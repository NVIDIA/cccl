// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_reduce.cuh>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
struct policy_hub_t
{
  _CCCL_API constexpr auto operator()(int /*arch*/) const -> ::cub::reduce_arch_policy
  {
    const auto policy = cub::agent_reduce_policy{
      TUNE_THREADS_PER_BLOCK,
      TUNE_ITEMS_PER_THREAD,
      1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS,
      cub::LOAD_DEFAULT};
    return {policy, policy, policy, policy};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using init_t   = T;

  // Retrieve axis parameters
  const auto elements = static_cast<offset_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // Allocate temporary storage:
  caching_allocator_t alloc;
  (void) cub::DeviceReduce::Reduce(
    d_in,
    d_out,
    elements,
    op_t{},
    init_t{},
    ::cuda::std::execution::env{::cuda::stream_ref{::cudaStream_t{nullptr}}, alloc});

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    (void) cub::DeviceReduce::Reduce(
      d_in,
      d_out,
      elements,
      op_t{},
      init_t{},
      ::cuda::std::execution::env{
        ::cuda::stream_ref{launch.get_stream().get_stream()},
        alloc
#if !TUNE_BASE
        ,
        ::cuda::std::execution::prop{::cuda::std::execution::__get_tuning_t, policy_hub_t{}}
#endif
      });
  });
}

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
