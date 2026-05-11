// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_reduce.cuh>

#include <cuda/__device/all_devices.h>
#include <cuda/__memory_pool/device_memory_pool.h>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
template <typename AccumT>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::reduce::reduce_policy
  {
    const auto [items, threads] =
      cub::detail::scale_mem_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, int{sizeof(AccumT)});
    const auto policy = cub::detail::reduce::agent_reduce_policy{
      threads, items, 1 << TUNE_ITEMS_PER_VEC_LOAD_POW2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy, policy};
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

  // So for now, we have to call into the dispatcher again to override the accumulator type:
  auto transform_op = ::cuda::std::identity{};

  std::size_t temp_size;
  cub::detail::reduce::dispatch(
    nullptr,
    temp_size,
    d_in,
    d_out,
    elements,
    op_t{},
    init_t{},
    nullptr /* stream */,
    transform_op
#if !TUNE_BASE
    ,
    policy_selector<T>{}
#endif
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::reduce::dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      elements,
      op_t{},
      init_t{},
      launch.get_stream(),
      transform_op
#if !TUNE_BASE
      ,
      policy_selector<T>{}
#endif
    );
  });
}

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
