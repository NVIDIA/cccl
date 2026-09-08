// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>

#include <cuda/cmath>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

__host__ __device__ __forceinline__ constexpr bool skip(int batches, int logical_warp_threads) noexcept
{
  return batches > logical_warp_threads || 4 * batches <= logical_warp_threads;
}

template <int LogicalWarpThreads, bool ToBlocked>
struct benchmark_batched_op_t
{
  template <typename T, cuda::std::size_t Batches>
  __device__ __forceinline__ cuda::std::array<T, Batches> operator()(cuda::std::array<T, Batches> thread_data) const
  {
    if constexpr (!skip(Batches, LogicalWarpThreads))
    {
#if 0
    using WarpReduceBatched           = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
    using TempStorage                 = typename WarpReduceBatched::TempStorage;
    constexpr auto max_out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
    cuda::std::array<T, max_out_per_thread> outputs;
    __shared__ TempStorage temp_storage;

    if constexpr (ToBlocked) {
      WarpReduceBatched{temp_storage}.ReduceToBlocked(thread_data, outputs, op_t{});
    }
    else
    {
      WarpReduceBatched{temp_storage}.ReduceToStriped(thread_data, outputs, op_t{});
    }

#  pragma unroll
    for (int i = 0; i < max_out_per_thread; ++i)
    {
      thread_data[i] = outputs[i];
    }
#else
      using WarpReduce  = cub::WarpReduce<T, LogicalWarpThreads>;
      using TempStorage = typename WarpReduce::TempStorage;
      __shared__ TempStorage temp_storage;

      WarpReduce warp_reduce{temp_storage};

// Sequentially reduce Batches arrays
#  pragma unroll
      for (int i = 0; i < Batches; ++i)
      {
        // This is somewhat of an unfair comparison since all results are returned by lane 0
        // while they are distributed among threads for WarpReduceBatched.
        thread_data[i] = warp_reduce.Reduce(thread_data[i], op_t{});
      }
#endif
    }
    return thread_data;
  }
};

template <int LogicalWarpThreads>
struct benchmark_sequential_op_t
{
  template <typename T, cuda::std::size_t Batches>
  __device__ __forceinline__ cuda::std::array<T, Batches> operator()(cuda::std::array<T, Batches> thread_data) const
  {
    return thread_data;
  }
};

using batches_list              = nvbench::enum_type_list<2, 3, 4, 8, 9, 12, 16, 23, 32>;
using logical_warp_threads_list = nvbench::enum_type_list<2, 4, 8, 16, 32>;
using to_blocked_list           = nvbench::enum_type_list<true, false>;

template <typename T, nvbench::int32_t Batches, nvbench::int32_t LogicalWarpThreads, bool ToBlocked>
void warp_reduce_batched(
  nvbench::state& state,
  nvbench::
    type_list<T, nvbench::enum_type<Batches>, nvbench::enum_type<LogicalWarpThreads>, nvbench::enum_type<ToBlocked>>)
{
  if constexpr (skip(Batches, LogicalWarpThreads))
  {
    state.skip("Skipping to avoid explosion of compile time");
    return;
  }
  constexpr int block_size                = 256;
  constexpr int min_reductions_per_thread = 128;
  constexpr int grid_size                 = (1 << 28) / (min_reductions_per_thread * block_size);
  constexpr int unroll_factor = cuda::ceil_div(min_reductions_per_thread, Batches); // Balance compile time and
                                                                                    // performance
  const auto& kernel =
    benchmark_kernel<block_size,
                     unroll_factor,
                     benchmark_batched_op_t<LogicalWarpThreads, ToBlocked>,
                     cuda::std::array<T, Batches>>;

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launcher) {
    kernel<<<grid_size, block_size, 0, launcher.get_stream()>>>(
      benchmark_batched_op_t<LogicalWarpThreads, ToBlocked>{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_batched,
                    NVBENCH_TYPE_AXES(value_types, batches_list, logical_warp_threads_list, to_blocked_list))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "Batches", "LogicalWarpThreads", "ToBlocked"});

template <typename T, nvbench::int32_t Batches, nvbench::int32_t LogicalWarpThreads, bool ToBlocked>
void warp_reduce_batched_latency(
  nvbench::state& state,
  nvbench::
    type_list<T, nvbench::enum_type<Batches>, nvbench::enum_type<LogicalWarpThreads>, nvbench::enum_type<ToBlocked>>)
{
  if constexpr (skip(Batches, LogicalWarpThreads))
  {
    state.skip("Skipping to avoid explosion of compile time");
    return;
  }
  constexpr int block_size    = cub::detail::warp_threads;
  constexpr int grid_size     = 1;
  constexpr int unroll_factor = cuda::ceil_div(256, Batches); // Balance compile time and
                                                              // performance
  const auto& kernel =
    benchmark_kernel<block_size,
                     unroll_factor,
                     benchmark_batched_op_t<LogicalWarpThreads, ToBlocked>,
                     cuda::std::array<T, Batches>>;

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launcher) {
    kernel<<<grid_size, block_size, 0, launcher.get_stream()>>>(
      benchmark_batched_op_t<LogicalWarpThreads, ToBlocked>{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_batched_latency,
                    NVBENCH_TYPE_AXES(value_types, batches_list, logical_warp_threads_list, to_blocked_list))
  .set_name("base-latency")
  .set_type_axes_names({"T{ct}", "Batches", "LogicalWarpThreads", "ToBlocked"});
