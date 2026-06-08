// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_broadcast.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

constexpr int block_size = 256;

struct manual_broadcast_t
{
  template <int LogicalWarpThreads, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T run(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;

    __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

    const int block_warp_id    = static_cast<int>(threadIdx.x) / LogicalWarpThreads;
    const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const T aggregate          = warp_reduce_t{temp_storage[block_warp_id]}.Sum(thread_data);
    const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
    return cub::ShuffleIndex<LogicalWarpThreads>(aggregate, 0, member_mask);
  }
};

struct primitive_broadcast_t
{
  template <int LogicalWarpThreads, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T run(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBroadcast<T, LogicalWarpThreads>;

    __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

    const int block_warp_id = static_cast<int>(threadIdx.x) / LogicalWarpThreads;
    return warp_reduce_t{temp_storage[block_warp_id]}.Sum(thread_data);
  }
};

template <typename BroadcastVariant, int LogicalWarpThreads>
struct broadcast_action_t
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    return BroadcastVariant{}.template run<LogicalWarpThreads>(thread_data);
  }
};

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

template <typename BroadcastVariant, typename T, typename LogicalWarpThreadsT>
void warp_reduce_broadcast(nvbench::state& state, nvbench::type_list<BroadcastVariant, T, LogicalWarpThreadsT>)
{
  constexpr int unroll_factor        = 128; // compromise between compile time and noise
  constexpr int logical_warp_threads = LogicalWarpThreadsT::value;
  using action_t                     = broadcast_action_t<BroadcastVariant, logical_warp_threads>;

  const auto& kernel    = benchmark_kernel<block_size, unroll_factor, action_t, T>;
  const int num_sms     = state.get_device().value().get_number_of_sms();
  const int device      = state.get_device().value().get_id();
  int max_blocks_per_sm = 0;

  NVBENCH_CUDA_CALL_NOEXCEPT(cudaSetDevice(device));
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_size, 0));

  const int grid_size = max_blocks_per_sm * num_sms;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(action_t{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_broadcast, NVBENCH_TYPE_AXES(broadcast_variants, value_types, logical_warp_sizes))
  .set_name("base")
  .set_type_axes_names({"Variant{ct}", "T{ct}", "LogicalWarpThreads{ct}"});
