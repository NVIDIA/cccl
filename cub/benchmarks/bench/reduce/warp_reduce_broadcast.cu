// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_broadcast.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

enum class benchmark_mode
{
  occupancy,
  fixed,
  // Used by the separate latency benchmark, not by the BenchmarkMode axis.
  latency,
  full_bounds,
};

benchmark_mode parse_benchmark_mode(const std::string& name)
{
  if (name == "occupancy")
  {
    return benchmark_mode::occupancy;
  }
  if (name == "fixed")
  {
    return benchmark_mode::fixed;
  }
  if (name == "full_bounds")
  {
    return benchmark_mode::full_bounds;
  }

  throw std::runtime_error("Unsupported BenchmarkMode axis value");
}

struct manual_broadcast_t
{
  template <int BlockThreads, int LogicalWarpThreads, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T run(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;

    __shared__ typename warp_reduce_t::TempStorage temp_storage[BlockThreads / LogicalWarpThreads];

    const int block_warp_id    = static_cast<int>(threadIdx.x) / LogicalWarpThreads;
    const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const T aggregate          = warp_reduce_t{temp_storage[block_warp_id]}.Sum(thread_data);
    const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
    return cub::ShuffleIndex<LogicalWarpThreads>(aggregate, 0, member_mask);
  }
};

struct primitive_broadcast_t
{
  template <int BlockThreads, int LogicalWarpThreads, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T run(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBroadcast<T, LogicalWarpThreads>;

    __shared__ typename warp_reduce_t::TempStorage temp_storage[BlockThreads / LogicalWarpThreads];

    const int block_warp_id = static_cast<int>(threadIdx.x) / LogicalWarpThreads;
    return warp_reduce_t{temp_storage[block_warp_id]}.Sum(thread_data);
  }
};

template <typename BroadcastVariant, int BlockThreads, int LogicalWarpThreads>
struct broadcast_action_t
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    return BroadcastVariant{}.template run<BlockThreads, LogicalWarpThreads>(thread_data);
  }
};

template <std::size_t Numerator, std::size_t Denominator>
inline constexpr std::size_t static_ceil_div = (Numerator + Denominator - 1) / Denominator;

template <int BlockThreads, int SmBlocks, int UnrollFactor, typename ActionT, typename T>
__launch_bounds__(BlockThreads, SmBlocks) __global__
  static void benchmark_kernel_full_bounds(_CCCL_GRID_CONSTANT const ActionT action)
{
  auto data = generate_random_data<T>();
  cuda::static_for<UnrollFactor>([&]([[maybe_unused]] auto _) {
    data = action(data);
  });
  sink(data);
}

using ::cuda::std::integral_constant;

using broadcast_variants = nvbench::type_list<manual_broadcast_t, primitive_broadcast_t>;
// This benchmark targets the integral Sum fast path; non-integral and generic fallbacks intentionally match
// WarpReduce plus an explicit broadcast.
using value_types =
  nvbench::type_list<::cuda::std::uint8_t,
                     ::cuda::std::int8_t,
                     ::cuda::std::uint16_t,
                     ::cuda::std::int16_t,
                     ::cuda::std::uint32_t,
                     ::cuda::std::int32_t,
                     ::cuda::std::uint64_t,
                     ::cuda::std::int64_t>;
using logical_warp_sizes =
  nvbench::type_list<integral_constant<int, 2>,
                     integral_constant<int, 4>,
                     integral_constant<int, 8>,
                     integral_constant<int, 16>,
                     integral_constant<int, 32>>;

template <int BlockThreads, benchmark_mode Mode, typename BroadcastVariant, typename T, typename LogicalWarpThreadsT>
void run_warp_reduce_broadcast(nvbench::state& state)
{
  constexpr int unroll_factor        = 128; // compromise between compile time and noise
  constexpr int logical_warp_threads = LogicalWarpThreadsT::value;
  using action_t                     = broadcast_action_t<BroadcastVariant, BlockThreads, logical_warp_threads>;

  const auto& kernel = benchmark_kernel<BlockThreads, unroll_factor, action_t, T>;

  if constexpr (Mode == benchmark_mode::latency)
  {
    constexpr int grid_size = 1;
    state.add_element_count(static_cast<std::size_t>(grid_size) * BlockThreads * unroll_factor);
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_size, BlockThreads, 0, launch.get_stream()>>>(action_t{});
    });
    return;
  }
  else
  {
    constexpr std::size_t fixed_work_items = std::size_t{1} << 28;
    constexpr std::size_t items_per_block  = BlockThreads * unroll_factor;
    constexpr int fixed_grid_size          = static_cast<int>(static_ceil_div<fixed_work_items, items_per_block>);

    if constexpr (Mode == benchmark_mode::full_bounds)
    {
      // Use a launch-bound target that is valid for the oldest SMs in the benchmark matrix.
      constexpr int resident_threads_per_sm   = 1024;
      constexpr int full_bounds_max_sm_blocks = resident_threads_per_sm / BlockThreads;
      const auto& full_bounds_kernel =
        benchmark_kernel_full_bounds<BlockThreads, full_bounds_max_sm_blocks, unroll_factor, action_t, T>;

      state.add_element_count(static_cast<std::size_t>(fixed_grid_size) * BlockThreads * unroll_factor);
      state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
        full_bounds_kernel<<<fixed_grid_size, BlockThreads, 0, launch.get_stream()>>>(action_t{});
      });
      return;
    }

    int grid_size = fixed_grid_size;

    if constexpr (Mode == benchmark_mode::occupancy)
    {
      const int num_sms     = state.get_device().value().get_number_of_sms();
      const int device      = state.get_device().value().get_id();
      int max_blocks_per_sm = 0;

      NVBENCH_CUDA_CALL_NOEXCEPT(cudaSetDevice(device));
      NVBENCH_CUDA_CALL_NOEXCEPT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, BlockThreads, 0));

      grid_size = max_blocks_per_sm * num_sms;
    }

    state.add_element_count(static_cast<std::size_t>(grid_size) * BlockThreads * unroll_factor);
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_size, BlockThreads, 0, launch.get_stream()>>>(action_t{});
    });
  }
}

template <typename BroadcastVariant, typename T, typename LogicalWarpThreadsT>
void warp_reduce_broadcast(nvbench::state& state, nvbench::type_list<BroadcastVariant, T, LogicalWarpThreadsT>)
{
  constexpr int block_threads = 256;
  const auto mode             = parse_benchmark_mode(state.get_string("BenchmarkMode"));

  switch (mode)
  {
    case benchmark_mode::occupancy:
      run_warp_reduce_broadcast<block_threads, benchmark_mode::occupancy, BroadcastVariant, T, LogicalWarpThreadsT>(
        state);
      return;
    case benchmark_mode::fixed:
      run_warp_reduce_broadcast<block_threads, benchmark_mode::fixed, BroadcastVariant, T, LogicalWarpThreadsT>(state);
      return;
    case benchmark_mode::full_bounds:
      run_warp_reduce_broadcast<block_threads, benchmark_mode::full_bounds, BroadcastVariant, T, LogicalWarpThreadsT>(
        state);
      return;
    case benchmark_mode::latency:
      break;
  }

  throw std::runtime_error("Unsupported BenchmarkMode axis value");
}

template <typename BroadcastVariant, typename T, typename LogicalWarpThreadsT>
void warp_reduce_broadcast_latency(nvbench::state& state, nvbench::type_list<BroadcastVariant, T, LogicalWarpThreadsT>)
{
  run_warp_reduce_broadcast<cub::detail::warp_threads, benchmark_mode::latency, BroadcastVariant, T, LogicalWarpThreadsT>(
    state);
}

NVBENCH_BENCH_TYPES(warp_reduce_broadcast, NVBENCH_TYPE_AXES(broadcast_variants, value_types, logical_warp_sizes))
  .set_name("base")
  .set_type_axes_names({"Variant{ct}", "T{ct}", "LogicalWarpThreads{ct}"})
  .add_string_axis("BenchmarkMode", {"occupancy", "fixed", "full_bounds"});

NVBENCH_BENCH_TYPES(warp_reduce_broadcast_latency,
                    NVBENCH_TYPE_AXES(broadcast_variants, value_types, logical_warp_sizes))
  .set_name("latency")
  .set_type_axes_names({"Variant{ct}", "T{ct}", "LogicalWarpThreads{ct}"});
