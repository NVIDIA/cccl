// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/thread/thread_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched_broadcast.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/std/array>
#include <cuda/std/functional>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

constexpr int block_size = 256;

template <int Batches, int LogicalWarpThreads, bool SyncPhysicalWarp = false>
struct manual_batched_broadcast_t
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    const int lane_id        = cub::detail::logical_lane_id<LogicalWarpThreads>();
    unsigned int member_mask = 0xFFFFFFFFu;
    if constexpr (!SyncPhysicalWarp)
    {
      const int logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
      member_mask               = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
    }

    ::cuda::std::array<T, Batches> outputs{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      outputs[batch] = thread_data + static_cast<T>(batch);
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = LogicalWarpThreads / 2; offset > 0; offset >>= 1)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int batch = 0; batch < Batches; ++batch)
      {
        outputs[batch] += cub::ShuffleIndex<LogicalWarpThreads>(outputs[batch], lane_id ^ offset, member_mask);
      }
    }

    return cub::ThreadReduce(outputs, ::cuda::std::plus<>{});
  }
};

template <int Batches, int LogicalWarpThreads, bool SyncPhysicalWarp = false>
struct primitive_batched_broadcast_t
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBatchedBroadcast<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;

    typename warp_reduce_t::TempStorage temp_storage;

    ::cuda::std::array<T, Batches> inputs{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      inputs[batch] = thread_data + static_cast<T>(batch);
    }

    return cub::ThreadReduce(warp_reduce_t{temp_storage}.Sum(inputs), ::cuda::std::plus<>{});
  }
};

template <int Batches, int LogicalWarpThreads>
struct serial_warp_reduce_broadcast_t
{
  template <typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;

    // This baseline intentionally models the straightforward user spelling:
    // invoke WarpReduce once per batch, then explicitly broadcast each owner-lane aggregate.
    // Its shared-memory footprint is therefore the cost of the serial CUB composition, not the new primitive.
    __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

    const int logical_block_warp_id = static_cast<int>(threadIdx.x) / LogicalWarpThreads;
    const int logical_warp_id       = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const auto member_mask          = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);

    T result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      const T aggregate = warp_reduce_t{temp_storage[logical_block_warp_id]}.Sum(thread_data + static_cast<T>(batch));
      result += cub::ShuffleIndex<LogicalWarpThreads>(aggregate, 0, member_mask);
    }
    return result;
  }
};

using broadcast_variants = nvbench::type_list<
  serial_warp_reduce_broadcast_t<1, 4>,
  manual_batched_broadcast_t<1, 4>,
  manual_batched_broadcast_t<1, 4, true>,
  primitive_batched_broadcast_t<1, 4>,
  primitive_batched_broadcast_t<1, 4, true>,
  serial_warp_reduce_broadcast_t<4, 4>,
  manual_batched_broadcast_t<4, 4>,
  manual_batched_broadcast_t<4, 4, true>,
  primitive_batched_broadcast_t<4, 4>,
  primitive_batched_broadcast_t<4, 4, true>,
  serial_warp_reduce_broadcast_t<5, 4>,
  manual_batched_broadcast_t<5, 4>,
  manual_batched_broadcast_t<5, 4, true>,
  primitive_batched_broadcast_t<5, 4>,
  primitive_batched_broadcast_t<5, 4, true>,
  serial_warp_reduce_broadcast_t<4, 8>,
  manual_batched_broadcast_t<4, 8>,
  manual_batched_broadcast_t<4, 8, true>,
  primitive_batched_broadcast_t<4, 8>,
  primitive_batched_broadcast_t<4, 8, true>,
  serial_warp_reduce_broadcast_t<4, 16>,
  manual_batched_broadcast_t<4, 16>,
  manual_batched_broadcast_t<4, 16, true>,
  primitive_batched_broadcast_t<4, 16>,
  primitive_batched_broadcast_t<4, 16, true>>;
using value_types = nvbench::type_list<int, float, double>;

template <typename BroadcastVariant, typename T>
void warp_reduce_batched_broadcast(nvbench::state& state, nvbench::type_list<BroadcastVariant, T>)
{
  constexpr int unroll_factor = 128; // compromise between compile time and noise
  const auto& kernel          = benchmark_kernel<block_size, unroll_factor, BroadcastVariant, T>;
  const int num_SMs           = state.get_device().value().get_number_of_sms();
  const int device            = state.get_device().value().get_id();
  int max_blocks_per_SM       = 0;

  NVBENCH_CUDA_CALL_NOEXCEPT(cudaSetDevice(device));
  NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_size, 0));

  const int grid_size = max_blocks_per_SM * num_SMs;
  if (grid_size == 0)
  {
    state.skip("Skipping: no resident blocks for this benchmark configuration.");
    return;
  }

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(BroadcastVariant{});
  });
}

NVBENCH_BENCH_TYPES(warp_reduce_batched_broadcast, NVBENCH_TYPE_AXES(broadcast_variants, value_types))
  .set_name("base")
  .set_type_axes_names({"Variant{ct}", "T{ct}"});
