// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#include <cub/block/block_reduce.cuh>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

template <int BlockThreads>
struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ T operator()(T thread_data) const
  {
    using BlockReduce = cub::BlockReduce<T, BlockThreads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    using TempStorage = typename BlockReduce::TempStorage;
    __shared__ TempStorage temp_storage;
    T agg = BlockReduce{temp_storage}.Reduce(thread_data, op_t{});
    // Re-broadcast so every thread depends on the reduction result, preventing DCE.
    __shared__ T broadcast;
    if (threadIdx.x == 0)
    {
      broadcast = agg;
    }
    __syncthreads();
    return broadcast;
  }
};

template <typename T>
void block_reduce_warp_reductions(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_size    = 256; // 8 warps -> exercises optimized ApplyWarpAggregates
  constexpr int unroll_factor = 32; // compromise between compile time and noise
  using action_t              = benchmark_op_t<block_size>;
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, action_t, T>;
  const int num_SMs           = state.get_device().value().get_number_of_sms();
  int max_blocks_per_SM       = 0;
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  const int grid_size = max_blocks_per_SM * num_SMs;
  if (grid_size == 0)
  {
    state.skip("Kernel occupancy is zero for this type/configuration.");
    return;
  }
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_size>>>(action_t{});
  });
}

NVBENCH_BENCH_TYPES(block_reduce_warp_reductions, NVBENCH_TYPE_AXES(value_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"});
