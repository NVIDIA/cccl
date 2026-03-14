// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>

#include <cuda/cmath>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

/**
 * @brief Benchmark operator for batched warp reduction
 *
 * Performs Batches reductions of LogicalWarpThreads elements each using WarpReduceBatched
 */
template <int LogicalWarpThreads>
struct benchmark_batched_op_t
{
  template <typename T, cuda::std::size_t Batches>
  __device__ __forceinline__ cuda::std::array<T, Batches> operator()(cuda::std::array<T, Batches> thread_data) const
  {
    using WarpReduceBatched           = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
    using TempStorage                 = typename WarpReduceBatched::TempStorage;
    constexpr auto max_out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
    cuda::std::array<T, max_out_per_thread> outputs;
    __shared__ TempStorage temp_storage;

    WarpReduceBatched{temp_storage}.Reduce(thread_data, outputs, op_t{});

#pragma unroll
    for (int i = 0; i < max_out_per_thread; ++i)
    {
      thread_data[i] = outputs[i];
    }
    return thread_data;
  }
};

/**
 * @brief Benchmark operator for sequential warp reductions (baseline)
 *
 * Performs Batches sequential calls to WarpReduce
 */
template <int LogicalWarpThreads>
struct benchmark_sequential_op_t
{
  template <typename T, cuda::std::size_t Batches>
  __device__ __forceinline__ cuda::std::array<T, Batches> operator()(cuda::std::array<T, Batches> thread_data) const
  {
    using WarpReduce  = cub::WarpReduce<T, LogicalWarpThreads>;
    using TempStorage = typename WarpReduce::TempStorage;
    __shared__ TempStorage temp_storage;

    WarpReduce warp_reduce{temp_storage};

// Sequentially reduce Batches arrays
#pragma unroll
    for (int i = 0; i < Batches; ++i)
    {
      // This is somewhat of an unfair comparison since all results are returned by lane 0
      thread_data[i] = warp_reduce.Reduce(thread_data[i], op_t{});
    }
    return thread_data;
  }
};

enum class launch_bounds_mode_t
{
  partial,
  full,
};

launch_bounds_mode_t parse_launch_bounds_mode(nvbench::state& state)
{
  const auto& launch_bounds_mode = state.get_string("LaunchBoundsMode");
  if (launch_bounds_mode == "partial")
  {
    return launch_bounds_mode_t::partial;
  }
  else if (launch_bounds_mode == "full")
  {
    return launch_bounds_mode_t::full;
  }
  else
  {
    throw std::invalid_argument("Invalid launch bounds mode: " + launch_bounds_mode);
  }
}

using batches_list              = nvbench::enum_type_list<8, 15, 16, 32>;
using logical_warp_threads_list = nvbench::enum_type_list<8, 16, 32>;

/**
 * @brief Run batched warp reduction benchmark
 */
template <typename T, nvbench::int32_t Batches, nvbench::int32_t LogicalWarpThreads>
void warp_reduce_batched(nvbench::state& state,
                         nvbench::type_list<T, nvbench::enum_type<Batches>, nvbench::enum_type<LogicalWarpThreads>>)
{
  constexpr int block_size                = 256;
  constexpr int max_sm_warps              = 48; // For consumer GPUs, datacenter allows 64, but same amount of registers
  constexpr int full_bounds_max_sm_blocks = (max_sm_warps * 32) / block_size;
  constexpr int unroll_factor = cuda::ceil_div(128, cuda::next_power_of_two(Batches)); // Balance compile time and
                                                                                       // performance
  const auto launch_bounds_mode = parse_launch_bounds_mode(state);
  const auto& kernel =
    launch_bounds_mode == launch_bounds_mode_t::full
      ? benchmark_kernel_full_bounds<block_size,
                                     full_bounds_max_sm_blocks,
                                     unroll_factor,
                                     benchmark_batched_op_t<LogicalWarpThreads>,
                                     cuda::std::array<T, Batches>>
      : benchmark_kernel<block_size,
                         unroll_factor,
                         benchmark_batched_op_t<LogicalWarpThreads>,
                         cuda::std::array<T, Batches>>;

  const int num_SMs = state.get_device().value().get_number_of_sms();

  int max_blocks_per_SM = 0;
  if (launch_bounds_mode == launch_bounds_mode_t::full)
  {
    max_blocks_per_SM = full_bounds_max_sm_blocks;
  }
  else
  {
    const int device = state.get_device().value().get_id();
    NVBENCH_CUDA_CALL_NOEXCEPT(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  }

  const int grid_size = max_blocks_per_SM * num_SMs;

  auto wspro_stages = 0;
  for (int stride_inter_reduce = 2; stride_inter_reduce <= LogicalWarpThreads; stride_inter_reduce *= 2)
  {
    wspro_stages += cuda::ceil_div(Batches, stride_inter_reduce);
  }

  // Add metadata
  state.add_element_count(grid_size * block_size * unroll_factor * Batches);
  state.add_summary("Stages").set_int64("value", wspro_stages);
  state.add_summary("Blocks").set_int64("value", grid_size);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launcher) {
    kernel<<<grid_size, block_size, 0, launcher.get_stream()>>>(benchmark_batched_op_t<LogicalWarpThreads>{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_batched, NVBENCH_TYPE_AXES(value_types, batches_list, logical_warp_threads_list))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "Batches", "LogicalWarpThreads"})
  .add_string_axis("LaunchBoundsMode", {"partial", "full"});

/**
 * @brief Run sequential warp reduction benchmark (baseline)
 */
template <typename T, nvbench::int32_t Batches, nvbench::int32_t LogicalWarpThreads>
void warp_reduce_sequential(nvbench::state& state,
                            nvbench::type_list<T, nvbench::enum_type<Batches>, nvbench::enum_type<LogicalWarpThreads>>)
{
  constexpr int block_size                = 256;
  constexpr int max_sm_warps              = 48; // For consumer GPUs, datacenter allows 64, but same amount of registers
  constexpr int full_bounds_max_sm_blocks = (max_sm_warps * 32) / block_size;
  constexpr int unroll_factor = cuda::ceil_div(128, cuda::next_power_of_two(Batches)); // Balance compile time and
                                                                                       // performance
  const auto launch_bounds_mode = parse_launch_bounds_mode(state);
  const auto& kernel =
    launch_bounds_mode == launch_bounds_mode_t::full
      ? benchmark_kernel_full_bounds<block_size,
                                     full_bounds_max_sm_blocks,
                                     unroll_factor,
                                     benchmark_sequential_op_t<LogicalWarpThreads>,
                                     cuda::std::array<T, Batches>>
      : benchmark_kernel<block_size,
                         unroll_factor,
                         benchmark_sequential_op_t<LogicalWarpThreads>,
                         cuda::std::array<T, Batches>>;

  const int num_SMs = state.get_device().value().get_number_of_sms();

  int max_blocks_per_SM = 0;
  if (launch_bounds_mode == launch_bounds_mode_t::full)
  {
    max_blocks_per_SM = full_bounds_max_sm_blocks;
  }
  else
  {
    const int device = state.get_device().value().get_id();
    NVBENCH_CUDA_CALL_NOEXCEPT(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));
  }

  const int grid_size          = max_blocks_per_SM * num_SMs;
  const auto sequential_stages = Batches * cuda::ilog2(LogicalWarpThreads);

  state.add_element_count(grid_size * block_size * unroll_factor * Batches);
  state.add_summary("Stages").set_int64("value", sequential_stages);
  state.add_summary("Blocks").set_int64("value", grid_size);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launcher) {
    kernel<<<grid_size, block_size, 0, launcher.get_stream()>>>(benchmark_sequential_op_t<LogicalWarpThreads>{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_sequential, NVBENCH_TYPE_AXES(value_types, batches_list, logical_warp_threads_list))
  .set_name("sequential")
  .set_type_axes_names({"T{ct}", "Batches", "LogicalWarpThreads"})
  .add_string_axis("LaunchBoundsMode", {"partial", "full"});
