// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! @rst
//! The ``cub::BlockRowReduce`` and ``cub::BlockRowReduceWarpBroadcast`` classes provide
//! :ref:`collective <collective-primitives>` methods for row-shaped block reductions.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/__functional/operations.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! Row-shaped block reduction for fixed layouts where each row spans one or more full warps.
//! The row aggregate is returned to every thread in the corresponding row. This matches common
//! norm kernels where a CTA owns one or more rows and every lane needs the row statistic.
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

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit BlockRowReduce(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(static_cast<int>(threadIdx.x))
  {}

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    const int warp_id     = linear_tid / WARP_THREADS;
    const int lane_id     = linear_tid % WARP_THREADS;
    const int row_id      = warp_id / WarpsPerRow;
    const int row_warp_id = warp_id % WarpsPerRow;

    T warp_aggregate = WarpReduceT(temp_storage.warp_reduce[warp_id]).Reduce(input, reduction_op);
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

      T row_aggregate = WarpReduceT(temp_storage.final_reduce[row_id]).Reduce(partial, reduction_op, WarpsPerRow);
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

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    return Reduce(input, ::cuda::std::plus<>{});
  }
};

//! @rst
//! Row-shaped block reduction that broadcasts the row aggregate by repeating the final
//! row-wide warp reduction in every warp of the row.
//!
//! This is intended for norm-style kernels where a CTA owns one or more rows, every
//! thread needs the row statistic, and ``WarpsPerRow`` fits in one warp. Compared to
//! ``BlockRowReduce``, this avoids storing the final row total and avoids the extra
//! CTA synchronizations needed to broadcast that stored total.
//! @endrst
template <typename T, int RowsPerBlock, int WarpsPerRow>
class BlockRowReduceWarpBroadcast
{
  static_assert(RowsPerBlock > 0, "RowsPerBlock must be greater than zero");
  static_assert(WarpsPerRow > 0, "WarpsPerRow must be greater than zero");
  static_assert(WarpsPerRow <= detail::warp_threads, "WarpsPerRow must fit in one final warp reduction");

  static constexpr int WARP_THREADS = detail::warp_threads;

  struct _TempStorage
  {
    cub::Uninitialized<T> partials[RowsPerBlock][WarpsPerRow];
  };

  _TempStorage& temp_storage;
  int linear_tid;

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T warp_all_reduce(T input, ReductionOp reduction_op) const
  {
    const int lane_id = linear_tid % WARP_THREADS;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = WARP_THREADS / 2; offset > 0; offset >>= 1)
    {
      const T peer = cub::ShuffleIndex<WARP_THREADS>(input, lane_id ^ offset, 0xFFFFFFFFu);
      input        = reduction_op(input, peer);
    }
    return input;
  }

public:
  /// @smemstorage{BlockRowReduceWarpBroadcast}
  struct TempStorage : cub::Uninitialized<_TempStorage>
  {};

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit BlockRowReduceWarpBroadcast(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(static_cast<int>(threadIdx.x))
  {}

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T CommutativeReduce(T input, ReductionOp reduction_op, T identity)
  {
    const int warp_id     = linear_tid / WARP_THREADS;
    const int lane_id     = linear_tid % WARP_THREADS;
    const int row_id      = warp_id / WarpsPerRow;
    const int row_warp_id = warp_id % WarpsPerRow;

    T warp_aggregate = warp_all_reduce(input, reduction_op);
    if (lane_id == 0)
    {
      temp_storage.partials[row_id][row_warp_id].Alias() = warp_aggregate;
    }
    __syncthreads();

    T partial = identity;
    if (lane_id < WarpsPerRow)
    {
      partial = temp_storage.partials[row_id][lane_id].Alias();
    }
    return warp_all_reduce(partial, reduction_op);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    return CommutativeReduce(input, ::cuda::std::plus<>{}, T{});
  }
};

CUB_NAMESPACE_END
