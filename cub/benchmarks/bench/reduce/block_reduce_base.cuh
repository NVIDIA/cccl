// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#include <cub/block/block_reduce.cuh>

#include <cuda/std/array>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

template <int BlockThreads, int ItemsPerThread>
struct benchmark_op_t
{
  template <typename T>
  __device__ __forceinline__ cuda::std::array<T, ItemsPerThread>
  operator()(cuda::std::array<T, ItemsPerThread> thread_data) const
  {
    using BlockReduce = cub::BlockReduce<T, BlockThreads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
    using TempStorage = typename BlockReduce::TempStorage;
    __shared__ TempStorage temp_storage;

    T items[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      items[i] = thread_data[i];
    }

    if constexpr (ItemsPerThread == 1)
    {
      thread_data[0] = BlockReduce{temp_storage}.Reduce(items[0], op_t{});
    }
    else
    {
      thread_data[0] = BlockReduce{temp_storage}.Reduce(items, op_t{});
    }
    return thread_data;
  }
};

template <int BlockThreads, int ItemsPerThread, typename T>
void block_reduce_warp_reductions_impl(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int unroll_factor = 32;
  constexpr int total_items   = 1 << 28; // large enough to hide tail effects on current devices
  using action_t              = benchmark_op_t<BlockThreads, ItemsPerThread>;
  using data_t                = cuda::std::array<T, ItemsPerThread>;
  const auto& kernel          = benchmark_kernel<BlockThreads, unroll_factor, action_t, data_t>;
  const int grid_size         = total_items / (BlockThreads * unroll_factor);
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, BlockThreads>>>(action_t{});
  });
}

template <int ItemsPerThread, typename T>
void block_reduce_warp_reductions_dispatch(nvbench::state& state, nvbench::type_list<T>)
{
  const int block_threads = static_cast<int>(state.get_int64("BlockThreads"));
  switch (block_threads)
  {
    case 128:
      block_reduce_warp_reductions_impl<128, ItemsPerThread>(state, nvbench::type_list<T>{});
      break;
    case 256:
      block_reduce_warp_reductions_impl<256, ItemsPerThread>(state, nvbench::type_list<T>{});
      break;
    case 512:
      block_reduce_warp_reductions_impl<512, ItemsPerThread>(state, nvbench::type_list<T>{});
      break;
    case 1024:
      block_reduce_warp_reductions_impl<1024, ItemsPerThread>(state, nvbench::type_list<T>{});
      break;
    default:
      state.skip("Unsupported BlockThreads axis value.");
      break;
  }
}

template <typename T>
void block_reduce_warp_reductions(nvbench::state& state, nvbench::type_list<T>)
{
  const int items_per_thread = static_cast<int>(state.get_int64("ItemsPerThread"));
  switch (items_per_thread)
  {
    case 1:
      block_reduce_warp_reductions_dispatch<1>(state, nvbench::type_list<T>{});
      break;
    case 4:
      block_reduce_warp_reductions_dispatch<4>(state, nvbench::type_list<T>{});
      break;
    case 16:
      block_reduce_warp_reductions_dispatch<16>(state, nvbench::type_list<T>{});
      break;
    default:
      state.skip("Unsupported ItemsPerThread axis value.");
      break;
  }
}

NVBENCH_BENCH_TYPES(block_reduce_warp_reductions, NVBENCH_TYPE_AXES(value_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_axis("BlockThreads", {128, 256, 512, 1024})
  .add_int64_axis("ItemsPerThread", {1, 4, 16});

template <typename T>
void block_reduce_warp_reductions_latency(nvbench::state& state, nvbench::type_list<T>)
{
  constexpr int block_threads = 32; // single warp to measure latency
  constexpr int unroll_factor = 32;
  using action_t              = benchmark_op_t<block_threads, 1>;
  using data_t                = cuda::std::array<T, 1>;
  const auto& kernel          = benchmark_kernel<block_threads, unroll_factor, action_t, data_t>;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<1, block_threads>>>(action_t{});
  });
}

NVBENCH_BENCH_TYPES(block_reduce_warp_reductions_latency, NVBENCH_TYPE_AXES(value_types))
  .set_name("latency")
  .set_type_axes_names({"T{ct}"});
