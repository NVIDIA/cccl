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
    return BlockReduce{temp_storage}.Reduce(thread_data, op_t{});
  }
};

template <int BlockThreads, typename T>
void block_reduce_warp_reductions_impl(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int unroll_factor = 32;
  constexpr int total_items   = 1 << 28; // large enough to hide tail effects on current devices
  using action_t              = benchmark_op_t<BlockThreads>;
  const auto& kernel          = benchmark_kernel<BlockThreads, unroll_factor, action_t, T>;
  const int grid_size         = total_items / (BlockThreads * unroll_factor);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, BlockThreads>>>(action_t{});
  });
}

template <typename T>
void block_reduce_warp_reductions(nvbench::state& state, nvbench::type_list<T>)
{
  const int block_threads = static_cast<int>(state.get_int64("BlockThreads"));
  switch (block_threads)
  {
    case 128:
      block_reduce_warp_reductions_impl<128>(state, nvbench::type_list<T>{});
      break;
    case 256:
      block_reduce_warp_reductions_impl<256>(state, nvbench::type_list<T>{});
      break;
    case 512:
      block_reduce_warp_reductions_impl<512>(state, nvbench::type_list<T>{});
      break;
    case 1024:
      block_reduce_warp_reductions_impl<1024>(state, nvbench::type_list<T>{});
      break;
    default:
      state.skip("Unsupported BlockThreads axis value.");
      break;
  }
}

NVBENCH_BENCH_TYPES(block_reduce_warp_reductions, NVBENCH_TYPE_AXES(value_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BlockThreads", {128, 256, 512, 1024});

template <typename T>
void block_reduce_warp_reductions_latency(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_threads = 32; // single warp to measure latency
  constexpr int unroll_factor = 32;
  using action_t              = benchmark_op_t<block_threads>;
  const auto& kernel          = benchmark_kernel<block_threads, unroll_factor, action_t, T>;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<1, block_threads>>>(action_t{});
  });
}

NVBENCH_BENCH_TYPES(block_reduce_warp_reductions_latency, NVBENCH_TYPE_AXES(value_types))
  .set_name("latency")
  .set_type_axes_names({"T{ct}"});
