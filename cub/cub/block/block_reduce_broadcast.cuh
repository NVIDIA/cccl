// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! @rst
//! The ``cub::BlockReduceBroadcast`` class provides :ref:`collective <collective-primitives>` methods for computing
//! block-wide reductions whose aggregate is returned to every thread in the block.
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
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__type_traits/is_trivially_destructible.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``BlockReduceBroadcast`` class provides :ref:`collective <collective-primitives>` methods for computing
//! block-wide reductions whose aggregate is returned to every thread in the block.
//!
//! .. versionadded:: 3.5.0
//!    First appears in CUDA Toolkit 13.5.
//!
//! Overview
//! ++++++++
//!
//! - Supports all ``cub::BlockReduce`` algorithms; ``BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC`` is restricted to
//!   the ``Sum(input)`` and ``Sum(inputs)`` overloads.
//! - Preserves ``BlockReduce`` reduction ordering for the selected algorithm.
//! - Broadcasts the aggregate through temporary storage so every thread receives the same return value.
//! - This primitive is useful for kernels that need the aggregate in every thread and would otherwise call
//!   ``BlockReduce`` followed by an explicit shared-memory broadcast.
//! - The broadcast aggregate uses placement construction in shared storage and requires trivially destructible ``T``.
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - Reuses ``BlockReduce`` for the reduction phase.
//! - Adds one shared-memory store and two block barriers to make the aggregate available to all threads.
//! - Requires the same reduction temporary storage as ``BlockReduce`` plus one aggregate item.
//!
//! @blockcollective{BlockReduceBroadcast}
//!
//! Simple Example
//! ++++++++++++++
//!
//! .. code-block:: c++
//!
//!    using BlockReduceBroadcast = cub::BlockReduceBroadcast<int, 128>;
//!    __shared__ typename BlockReduceBroadcast::TempStorage temp_storage;
//!
//!    int thread_data = ...;
//!    int block_sum = BlockReduceBroadcast(temp_storage).Sum(thread_data);
//!    // block_sum is valid in every thread in the block.
//!
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type. Must be trivially destructible because the broadcast aggregate is
//!   placement-constructed in temporary shared storage that may be reused by later collective calls.
//!
//! @tparam BlockDimX
//!   The thread block length in threads along the X dimension.
//!
//! @tparam Algorithm
//!   **[optional]** The ``cub::BlockReduceAlgorithm`` specialization to use
//!   (default: ``cub::BLOCK_REDUCE_WARP_REDUCTIONS``).
//!
//! @tparam BlockDimY
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1).
//!
//! @tparam BlockDimZ
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1).
template <typename T,
          int BlockDimX,
          BlockReduceAlgorithm Algorithm = BLOCK_REDUCE_WARP_REDUCTIONS,
          int BlockDimY                  = 1,
          int BlockDimZ                  = 1>
class BlockReduceBroadcast
{
  using BlockReduceT                        = BlockReduce<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;
  static constexpr bool IS_NONDETERMINISTIC = Algorithm == BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;

  static_assert(::cuda::std::is_trivially_destructible_v<T>,
                "BlockReduceBroadcast requires a trivially destructible value type because it reuses shared "
                "TempStorage without destroying previous broadcast aggregates");

  template <bool IsSupported>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static constexpr void AssertAlgorithmSupportsOverload()
  {
    static_assert(IsSupported,
                  "BlockReduceBroadcast supports BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC only for Sum(input) "
                  "and Sum(inputs) overloads; Reduce and partial Sum overloads are unsupported");
  }

  struct _TempStorage
  {
    typename BlockReduceT::TempStorage reduce;
    Uninitialized<T> aggregate;
  };

  //! Shared storage reference.
  _TempStorage& temp_storage;

  //! Linear thread id in row-major order.
  unsigned int linear_tid;

  //! Internal storage allocator.
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T BroadcastAggregate(T aggregate)
  {
    if (linear_tid == 0)
    {
      detail::uninitialized_copy_single(&temp_storage.aggregate.Alias(), aggregate);
    }
    __syncthreads();

    const T result = temp_storage.aggregate.Alias();
    __syncthreads();
    return result;
  }

public:
  //! @smemstorage{BlockReduceBroadcast}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using a private static allocation of shared memory as temporary storage.
  //! @endrst
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceBroadcast()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockReduceBroadcast(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @}
  //! @name Generic reductions
  //! @{

  //! @rst
  //! Computes a block-wide reduction using the specified binary reduction functor and returns the aggregate to every
  //! thread in the block.
  //!
  //! @smemreuse
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator.
  //!
  //! @return
  //!   The block-wide reduction aggregate, returned in every thread in the block.
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    AssertAlgorithmSupportsOverload<!IS_NONDETERMINISTIC>();
    return BroadcastAggregate(BlockReduceT{temp_storage.reduce}.Reduce(input, reduction_op));
  }

  //! @rst
  //! Computes a block-wide reduction of multiple items per thread in a blocked arrangement and returns the aggregate to
  //! every thread in the block.
  //!
  //! - @granularity
  //! @smemreuse
  //! @endrst
  //!
  //! @tparam ItemsPerThread
  //!   **[inferred]** The number of input items per thread.
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] inputs
  //!   Calling thread's input items.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator.
  //!
  //! @return
  //!   The block-wide reduction aggregate, returned in every thread in the block.
  template <int ItemsPerThread, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T (&inputs)[ItemsPerThread], ReductionOp reduction_op)
  {
    AssertAlgorithmSupportsOverload<!IS_NONDETERMINISTIC>();
    return BroadcastAggregate(BlockReduceT{temp_storage.reduce}.Reduce(inputs, reduction_op));
  }

  //! @rst
  //! Computes a block-wide reduction of up to ``num_valid`` threads and returns the aggregate to every thread in the
  //! block.
  //!
  //! @smemreuse
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Binary reduction operator type having member ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @param[in] reduction_op
  //!   Binary reduction operator.
  //!
  //! @param[in] num_valid
  //!   Number of threads containing valid elements (may be less than the number of threads in the block).
  //!
  //! @return
  //!   The block-wide reduction aggregate, returned in every thread in the block.
  template <typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, int num_valid)
  {
    AssertAlgorithmSupportsOverload<!IS_NONDETERMINISTIC>();
    return BroadcastAggregate(BlockReduceT{temp_storage.reduce}.Reduce(input, reduction_op, num_valid));
  }

  //! @}
  //! @name Sum reductions
  //! @{

  //! @rst
  //! Computes a block-wide sum of one item per thread and returns the aggregate to every thread in the block.
  //!
  //! @smemreuse
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @return
  //!   The block-wide sum, returned in every thread in the block.
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input)
  {
    return BroadcastAggregate(BlockReduceT{temp_storage.reduce}.Sum(input));
  }

  //! @rst
  //! Computes a block-wide sum of multiple items per thread in a blocked arrangement and returns the aggregate to every
  //! thread in the block.
  //!
  //! - @granularity
  //! @smemreuse
  //! @endrst
  //!
  //! @tparam ItemsPerThread
  //!   **[inferred]** The number of input items per thread.
  //!
  //! @param[in] inputs
  //!   Calling thread's input items.
  //!
  //! @return
  //!   The block-wide sum, returned in every thread in the block.
  template <int ItemsPerThread>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T (&inputs)[ItemsPerThread])
  {
    return BroadcastAggregate(BlockReduceT{temp_storage.reduce}.Sum(inputs));
  }

  //! @rst
  //! Computes a block-wide sum of up to ``num_valid`` threads and returns the aggregate to every thread in the block.
  //!
  //! - This overload does not support the ``BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC`` algorithm. Use a full-tile
  //!   ``Sum`` overload with that algorithm.
  //!
  //! @smemreuse
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @param[in] num_valid
  //!   Number of threads containing valid elements (may be less than the number of threads in the block).
  //!
  //! @return
  //!   The sum of the valid items, returned in every thread in the block.
  _CCCL_DEVICE _CCCL_FORCEINLINE T Sum(T input, int num_valid)
  {
    AssertAlgorithmSupportsOverload<!IS_NONDETERMINISTIC>();
    return BroadcastAggregate(BlockReduceT{temp_storage.reduce}.Sum(input, num_valid));
  }

  //! @}
};

CUB_NAMESPACE_END
