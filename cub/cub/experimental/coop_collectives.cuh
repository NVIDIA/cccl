// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! @rst
//! Experimental CUB collective adapters with result-placement semantics that are useful for
//! cooperative-library frontends.
//! @endrst

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__functional/operations.h>

CUB_NAMESPACE_BEGIN

namespace experimental
{
//! @rst
//! Warp-wide reduction adapter that broadcasts the aggregate to every participating logical lane.
//! This composes CUB's ``WarpReduce`` owner-lane reduction with a warp shuffle broadcast.
//! @endrst
template <typename T, int LogicalWarpThreads = detail::warp_threads>
class WarpReduceBroadcast
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");

  using WarpReduceT = cub::WarpReduce<T, LogicalWarpThreads>;

  typename WarpReduceT::TempStorage& temp_storage;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T BroadcastFromLane0(T aggregate) const
  {
    const auto logical_warp_id = cub::detail::logical_warp_id<LogicalWarpThreads>();
    const auto member_mask     = cub::WarpMask<LogicalWarpThreads>(logical_warp_id);
    return cub::ShuffleIndex<LogicalWarpThreads>(aggregate, 0, member_mask);
  }

public:
  /// @smemstorage{WarpReduceBroadcast}
  using TempStorage = typename WarpReduceT::TempStorage;

  _CCCL_DEVICE _CCCL_FORCEINLINE explicit WarpReduceBroadcast(TempStorage& temp_storage)
      : temp_storage(temp_storage)
  {}

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    return BroadcastFromLane0(WarpReduceT(temp_storage).Sum(input));
  }

  _CCCL_TEMPLATE(typename InputType)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputType>)
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(const InputType& input)
  {
    return BroadcastFromLane0(WarpReduceT(temp_storage).Sum(input));
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int valid_items)
  {
    return BroadcastFromLane0(WarpReduceT(temp_storage).Sum(input, valid_items));
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    return BroadcastFromLane0(WarpReduceT(temp_storage).Reduce(input, reduction_op));
  }

  _CCCL_TEMPLATE(typename InputType, typename ReductionOp)
  _CCCL_REQUIRES(detail::is_fixed_size_random_access_range_v<InputType>)
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(const InputType& input, ReductionOp reduction_op)
  {
    return BroadcastFromLane0(WarpReduceT(temp_storage).Reduce(input, reduction_op));
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int valid_items)
  {
    return BroadcastFromLane0(WarpReduceT(temp_storage).Reduce(input, reduction_op, valid_items));
  }
};

//! @rst
//! Block-wide reduction adapter that broadcasts the aggregate to every thread in the block.
//! This keeps the usual CUB ``BlockReduce`` algorithm selection and stores the owner-lane result
//! in user-provided temporary storage before broadcasting it through shared memory.
//! @endrst
template <typename T,
          int BlockDimX,
          BlockReduceAlgorithm Algorithm = BLOCK_REDUCE_WARP_REDUCTIONS,
          int BlockDimY                  = 1,
          int BlockDimZ                  = 1>
class BlockReduceBroadcast
{
  static constexpr int BLOCK_THREADS = BlockDimX * BlockDimY * BlockDimZ;

  using BlockReduceT = cub::BlockReduce<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;

  struct _TempStorage
  {
    typename BlockReduceT::TempStorage reduce;
    cub::Uninitialized<T> aggregate;
  };

  _TempStorage& temp_storage;
  unsigned int linear_tid;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Broadcast(T aggregate)
  {
    if (linear_tid == 0)
    {
      temp_storage.aggregate.Alias() = aggregate;
    }
    __syncthreads();

    T result = temp_storage.aggregate.Alias();
    __syncthreads();
    return result;
  }

public:
  /// @smemstorage{BlockReduceBroadcast}
  struct TempStorage : cub::Uninitialized<_TempStorage>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE explicit BlockReduceBroadcast(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    return Broadcast(BlockReduceT(temp_storage.reduce).Reduce(input, reduction_op));
  }

  template <int ITEMS_PER_THREAD, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T (&inputs)[ITEMS_PER_THREAD], ReductionOp reduction_op)
  {
    return Broadcast(BlockReduceT(temp_storage.reduce).Reduce(inputs, reduction_op));
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int num_valid)
  {
    return Broadcast(BlockReduceT(temp_storage.reduce).Reduce(input, reduction_op, num_valid));
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    return Broadcast(BlockReduceT(temp_storage.reduce).Sum(input));
  }

  template <int ITEMS_PER_THREAD>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T (&inputs)[ITEMS_PER_THREAD])
  {
    return Broadcast(BlockReduceT(temp_storage.reduce).Sum(inputs));
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    return Broadcast(BlockReduceT(temp_storage.reduce).Sum(input, num_valid));
  }
};

//! @rst
//! Row-shaped block reduction for fixed layouts where each row spans one or more full warps.
//! The row sum is returned to every thread in the corresponding row. This matches common norm
//! kernels where a CTA owns one or more rows and every lane needs the row statistic.
//! @endrst
template <typename T, int RowsPerBlock, int WarpsPerRow>
class BlockRowReduce
{
  static_assert(RowsPerBlock > 0, "RowsPerBlock must be greater than zero");
  static_assert(WarpsPerRow > 0, "WarpsPerRow must be greater than zero");
  static_assert(WarpsPerRow <= detail::warp_threads, "WarpsPerRow must fit in one final warp reduction");

  static constexpr int WARP_THREADS  = detail::warp_threads;
  static constexpr int BLOCK_THREADS = RowsPerBlock * WarpsPerRow * WARP_THREADS;
  static constexpr int WARPS         = RowsPerBlock * WarpsPerRow;

  using WarpReduceT = cub::WarpReduce<T, WARP_THREADS>;

  struct _TempStorage
  {
    typename WarpReduceT::TempStorage warp_reduce[WARPS];
    typename WarpReduceT::TempStorage final_reduce[RowsPerBlock];
    cub::Uninitialized<T> partials[RowsPerBlock][WarpsPerRow];
    cub::Uninitialized<T> totals[RowsPerBlock];
  };

  _TempStorage& temp_storage;
  int linear_tid;

public:
  /// @smemstorage{BlockRowReduce}
  struct TempStorage : cub::Uninitialized<_TempStorage>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE explicit BlockRowReduce(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(static_cast<int>(threadIdx.x))
  {}

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    const int warp_id     = linear_tid / WARP_THREADS;
    const int lane_id     = linear_tid % WARP_THREADS;
    const int row_id      = warp_id / WarpsPerRow;
    const int row_warp_id = warp_id % WarpsPerRow;

    T warp_aggregate = WarpReduceT(temp_storage.warp_reduce[warp_id]).Sum(input);
    if (lane_id == 0)
    {
      temp_storage.partials[row_id][row_warp_id].Alias() = warp_aggregate;
    }
    __syncthreads();

    if (row_warp_id == 0)
    {
      T partial = T{};
      if (lane_id < WarpsPerRow)
      {
        partial = temp_storage.partials[row_id][lane_id].Alias();
      }

      T row_aggregate = WarpReduceT(temp_storage.final_reduce[row_id]).Sum(partial, WarpsPerRow);
      if (lane_id == 0)
      {
        temp_storage.totals[row_id].Alias() = row_aggregate;
      }
    }
    __syncthreads();

    T result = temp_storage.totals[row_id].Alias();
    __syncthreads();
    return result;
  }
};
} // namespace experimental

CUB_NAMESPACE_END
