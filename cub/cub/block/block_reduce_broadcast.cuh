// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! @rst
//! The ``cub::BlockReduceBroadcast`` class provides :ref:`collective <collective-primitives>` methods for
//! computing block-wide reductions whose aggregate is returned to every thread in the block.
//! @endrst

#pragma once

#include <cub/config.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

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
  using BlockReduceT = cub::BlockReduce<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;

  struct _TempStorage
  {
    typename BlockReduceT::TempStorage reduce;
    cub::Uninitialized<T> aggregate;
  };

  _TempStorage& temp_storage;
  unsigned int linear_tid;

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T broadcast(T aggregate)
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

  _CCCL_DEVICE_API _CCCL_FORCEINLINE explicit BlockReduceBroadcast(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    return broadcast(BlockReduceT(temp_storage.reduce).Reduce(input, reduction_op));
  }

  template <int ITEMS_PER_THREAD, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T (&inputs)[ITEMS_PER_THREAD], ReductionOp reduction_op)
  {
    return broadcast(BlockReduceT(temp_storage.reduce).Reduce(inputs, reduction_op));
  }

  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int num_valid)
  {
    return broadcast(BlockReduceT(temp_storage.reduce).Reduce(input, reduction_op, num_valid));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    return broadcast(BlockReduceT(temp_storage.reduce).Sum(input));
  }

  template <int ITEMS_PER_THREAD>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T (&inputs)[ITEMS_PER_THREAD])
  {
    return broadcast(BlockReduceT(temp_storage.reduce).Sum(inputs));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    return broadcast(BlockReduceT(temp_storage.reduce).Sum(input, num_valid));
  }
};

CUB_NAMESPACE_END
