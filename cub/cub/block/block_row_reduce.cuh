// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

//! @rst
//! The ``BlockRowReduce`` class provides :ref:`collective <collective-primitives>` methods for reducing a thread block
//! whose row-major linear thread ids are partitioned into fixed-size rows, where each row spans one or more full warps.
//!
//! Overview
//! ++++++++
//!
//! - A row contains ``WarpsPerRow`` consecutive architectural warps.
//! - A block contains ``RowsPerBlock`` independent rows.
//! - Each row aggregate is returned to every thread in the corresponding row.
//! - Generic ``Reduce`` supports associative reduction operators that need not be commutative. It uses
//!   ``cub::WarpReduce`` trees within each warp and across row partials; bit-exact left-to-right ordering is not
//!   guaranteed.
//! - This primitive is useful for norm-style kernels where a CTA owns one or more rows and every thread needs the row
//!   statistic.
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - Uses one warp-wide reduction per warp to form row partials.
//! - Single-warp rows broadcast the aggregate with a warp shuffle and use no block barriers.
//! - Multi-warp rows use one additional warp-wide reduction per row and two block barriers: one to publish per-warp
//!   partials, and one to broadcast the row aggregate through temporary storage.
//! - Row partials and row aggregates use temporary shared storage and warp shuffles, and require trivially copyable
//!   ``T``.
//!
//! .. versionadded:: 3.5.0
//!    First appears in CUDA Toolkit 13.5.
//!
//! @blockcollective{BlockRowReduce}
//!
//! Simple Example
//! ++++++++++++++
//!
//! .. code-block:: c++
//!
//!    using RowReduce = cub::BlockRowReduce<float, 1, 4>;
//!    __shared__ typename RowReduce::TempStorage temp_storage;
//!
//!    float row_sum = RowReduce(temp_storage).Sum(thread_data);
//!    // row_sum is valid in every thread in the row.
//!
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type. Must be trivially copyable because values are communicated with warp
//!   shuffles and stored in temporary shared storage that may be reused by later collective calls.
//!
//! @tparam RowsPerBlock
//!   The number of independent rows in the thread block.
//!
//! @tparam WarpsPerRow
//!   The number of full warps assigned to each row.
template <typename T, int RowsPerBlock, int WarpsPerRow>
class BlockRowReduce
{
  static_assert(RowsPerBlock > 0, "RowsPerBlock must be greater than zero");
  static_assert(WarpsPerRow > 0, "WarpsPerRow must be greater than zero");
  static_assert(WarpsPerRow <= detail::warp_threads, "WarpsPerRow must fit in one final warp reduction");

  static constexpr int warp_threads  = detail::warp_threads;
  static constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_threads;
  static constexpr int warps         = RowsPerBlock * WarpsPerRow;

  static_assert(block_threads <= 1024, "RowsPerBlock * WarpsPerRow * warp_threads must fit in one CUDA thread block");

  using WarpReduceT = WarpReduce<T, warp_threads>;

  static_assert(::cuda::std::is_trivially_copyable_v<T>,
                "BlockRowReduce requires a trivially copyable value type because it communicates values through warp "
                "shuffles and reuses shared TempStorage");

  struct _TempStorageSingleWarp
  {
    typename WarpReduceT::TempStorage warp_reduce[warps];
  };

  struct _TempStorageMultiWarp
  {
    // Keep phase fields explicit instead of overlaying partials and totals. The extra row-total slot avoids manual
    // union lifetime management while keeping the layout simple; full-warp WarpReduce storage is NullType.
    typename WarpReduceT::TempStorage warp_reduce[warps];
    typename WarpReduceT::TempStorage final_reduce[RowsPerBlock];
    Uninitialized<T> partials[RowsPerBlock][WarpsPerRow];
    Uninitialized<T> totals[RowsPerBlock];
  };

  using _TempStorage = ::cuda::std::conditional_t<WarpsPerRow == 1, _TempStorageSingleWarp, _TempStorageMultiWarp>;

  //! Shared storage reference.
  _TempStorage& temp_storage;

  //! Row-major linear thread id.
  int linear_tid;

  //! Internal storage allocator.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

public:
  //! @smemstorage{BlockRowReduce}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! Number of threads in the row-shaped block.
  static constexpr int BLOCK_THREADS = block_threads;

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using a private static allocation of shared memory as temporary storage.
  //! Thread participation is based on the row-major linear thread id.
  //! @endrst
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduce()
      : BlockRowReduce(RowMajorTid(blockDim.x, blockDim.y, blockDim.z))
  {}

  //! @rst
  //! Collective constructor using a private static allocation of shared memory as temporary storage.
  //! Thread participation is based on ``linear_tid``.
  //! @endrst
  //!
  //! @param[in] linear_tid Calling thread's row-major linear thread id
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduce(int linear_tid)
      : temp_storage(PrivateStorage())
      , linear_tid(linear_tid)
  {}

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Thread participation is based on the row-major linear thread id.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduce(TempStorage& temp_storage)
      : BlockRowReduce(temp_storage, RowMajorTid(blockDim.x, blockDim.y, blockDim.z))
  {}

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Thread participation is based on ``linear_tid``.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  //! @param[in] linear_tid Calling thread's row-major linear thread id
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduce(TempStorage& temp_storage, int linear_tid)
      : temp_storage(temp_storage.Alias())
      , linear_tid(linear_tid)
  {}

  //! @}
  //! @name Generic reductions
  //! @{

  //! @rst
  //! Computes a row-wide reduction using the specified binary reduction functor and returns the row aggregate to every
  //! thread in the row.
  //! The reduction operator must be associative. It need not be commutative.
  //!
  //! - @smemreuse
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
  //!   The row-wide reduction aggregate.
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op)
  {
    const int warp_id = linear_tid / warp_threads;

    const T warp_aggregate = WarpReduceT{temp_storage.warp_reduce[warp_id]}.Reduce(input, reduction_op);
    if constexpr (WarpsPerRow == 1)
    {
      // Rows are made from full architectural warps, so every lane participates in the broadcast.
      return ShuffleIndex<warp_threads>(warp_aggregate, 0, 0xFFFFFFFFu);
    }
    else
    {
      const int lane_id     = linear_tid % warp_threads;
      const int row_id      = warp_id / WarpsPerRow;
      const int row_warp_id = warp_id % WarpsPerRow;

      if (lane_id == 0)
      {
        detail::uninitialized_copy_single(&temp_storage.partials[row_id][row_warp_id].Alias(), warp_aggregate);
      }
      __syncthreads();

      if (row_warp_id == 0)
      {
        // WarpReduce's valid_items overload only consumes lanes [0, WarpsPerRow). Lanes outside the row partial range
        // still provide a valid object because T need not be default-constructible; only lane 0's aggregate is
        // consumed.
        T partial = input;
        if (lane_id < WarpsPerRow)
        {
          partial = temp_storage.partials[row_id][lane_id].Alias();
        }

        const T row_aggregate =
          WarpReduceT{temp_storage.final_reduce[row_id]}.Reduce(partial, reduction_op, WarpsPerRow);
        if (lane_id == 0)
        {
          detail::uninitialized_copy_single(&temp_storage.totals[row_id].Alias(), row_aggregate);
        }
      }
      __syncthreads();

      const T result = temp_storage.totals[row_id].Alias();
      return result;
    }
  }

  //! @}
  //! @name Sum reductions
  //! @{

  //! @rst
  //! Computes a row-wide sum and returns the row aggregate to every thread in the row.
  //!
  //! - @smemreuse
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @return
  //!   The row-wide sum.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    return Reduce(input, ::cuda::std::plus<>{});
  }

  //! @}
};

//! @rst
//! The ``BlockRowReduceWarpBroadcast`` class provides :ref:`collective <collective-primitives>` methods for reducing a
//! thread block whose row-major linear thread ids are partitioned into fixed-size rows, where each row spans one or
//! more full warps.
//!
//! Overview
//! ++++++++
//!
//! - A row contains ``WarpsPerRow`` consecutive architectural warps.
//! - A block contains ``RowsPerBlock`` independent rows.
//! - Each row aggregate is returned to every thread in the corresponding row.
//! - The reduction operator must be commutative and associative. Use ``BlockRowReduce`` for associative operators that
//!   do not commute.
//! - Instead of storing one final row total and broadcasting it with block barriers, every warp in the row repeats the
//!   final reduction over the row partials.
//!
//! Performance Characteristics
//! +++++++++++++++++++++++++++
//!
//! - Uses one warp-wide all-reduce for a single-warp row.
//! - For multi-warp rows, uses one warp-wide reduction per warp to form row partials.
//! - Uses one warp-wide all-reduce in every row warp to broadcast the row aggregate.
//! - The row-broadcast all-reduce uses ``log2(warp_threads)`` shuffle rounds, even when ``WarpsPerRow < warp_threads``.
//! - Avoids the final shared-memory row total and one block barrier used by ``BlockRowReduce``.
//! - Row partials use temporary shared storage and warp shuffles, and require trivially copyable ``T``.
//!
//! .. versionadded:: 3.5.0
//!    First appears in CUDA Toolkit 13.5.
//!
//! @blockcollective{BlockRowReduceWarpBroadcast}
//!
//! Simple Example
//! ++++++++++++++
//!
//! .. code-block:: c++
//!
//!    using RowReduce = cub::BlockRowReduceWarpBroadcast<float, 1, 4>;
//!    __shared__ typename RowReduce::TempStorage temp_storage;
//!
//!    float row_sum = RowReduce(temp_storage).Sum(thread_data);
//!    // row_sum is valid in every thread in the row.
//!
//! @endrst
//!
//! @tparam T
//!   The reduction input/output element type. Must be trivially copyable because values are communicated with warp
//!   shuffles and stored in temporary shared storage that may be reused by later collective calls.
//!
//! @tparam RowsPerBlock
//!   The number of independent rows in the thread block.
//!
//! @tparam WarpsPerRow
//!   The number of full warps assigned to each row.
template <typename T, int RowsPerBlock, int WarpsPerRow>
class BlockRowReduceWarpBroadcast
{
  static_assert(RowsPerBlock > 0, "RowsPerBlock must be greater than zero");
  static_assert(WarpsPerRow > 0, "WarpsPerRow must be greater than zero");
  static_assert(WarpsPerRow <= detail::warp_threads, "WarpsPerRow must fit in one final warp reduction");

  static constexpr int warp_threads  = detail::warp_threads;
  static constexpr int block_threads = RowsPerBlock * WarpsPerRow * warp_threads;
  static constexpr int warps         = RowsPerBlock * WarpsPerRow;

  static_assert(block_threads <= 1024, "RowsPerBlock * WarpsPerRow * warp_threads must fit in one CUDA thread block");

  using WarpReduceT = WarpReduce<T, warp_threads>;

  static_assert(::cuda::std::is_trivially_copyable_v<T>,
                "BlockRowReduceWarpBroadcast requires a trivially copyable value type because it communicates values "
                "through warp shuffles and reuses shared TempStorage");

  struct _TempStorageSingleWarp
  {};

  struct _TempStorageMultiWarp
  {
    typename WarpReduceT::TempStorage warp_reduce[warps];
    Uninitialized<T> partials[RowsPerBlock][WarpsPerRow];
  };

  using _TempStorage = ::cuda::std::conditional_t<WarpsPerRow == 1, _TempStorageSingleWarp, _TempStorageMultiWarp>;

  //! Shared storage reference.
  _TempStorage& temp_storage;

  //! Row-major linear thread id.
  int linear_tid;

  //! Internal storage allocator.
  _CCCL_DEVICE_API _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  template <typename ReductionOp>
  [[nodiscard]] static _CCCL_DEVICE_API _CCCL_FORCEINLINE T WarpAllReduce(T input, ReductionOp reduction_op, int lane_id)
  {
    // Unlike WarpReduce, which returns the aggregate to lane 0, this full-warp butterfly produces an all-reduce for
    // commutative operators so every lane can consume the row aggregate without a trailing broadcast. Each lane
    // traverses the same balanced partition tree up to commuted operands at each tree node; the commutative contract
    // makes the aggregate lane-invariant while avoiding an additional shuffle broadcast.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = warp_threads / 2; offset > 0; offset >>= 1)
    {
      const T peer = ShuffleIndex<warp_threads>(input, lane_id ^ offset, 0xFFFFFFFFu);
      input        = reduction_op(input, peer);
    }
    return input;
  }

public:
  //! @smemstorage{BlockRowReduceWarpBroadcast}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! Number of threads in the row-shaped block.
  static constexpr int BLOCK_THREADS = block_threads;

  //! @name Collective constructors
  //! @{

  //! @rst
  //! Collective constructor using a private static allocation of shared memory as temporary storage.
  //! Thread participation is based on the row-major linear thread id.
  //! @endrst
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduceWarpBroadcast()
      : BlockRowReduceWarpBroadcast(RowMajorTid(blockDim.x, blockDim.y, blockDim.z))
  {}

  //! @rst
  //! Collective constructor using a private static allocation of shared memory as temporary storage.
  //! Thread participation is based on ``linear_tid``.
  //! @endrst
  //!
  //! @param[in] linear_tid Calling thread's row-major linear thread id
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduceWarpBroadcast(int linear_tid)
      : temp_storage(PrivateStorage())
      , linear_tid(linear_tid)
  {}

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Thread participation is based on the row-major linear thread id.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduceWarpBroadcast(TempStorage& temp_storage)
      : BlockRowReduceWarpBroadcast(temp_storage, RowMajorTid(blockDim.x, blockDim.y, blockDim.z))
  {}

  //! @rst
  //! Collective constructor using the specified memory allocation as temporary storage.
  //! Thread participation is based on ``linear_tid``.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  //! @param[in] linear_tid Calling thread's row-major linear thread id
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockRowReduceWarpBroadcast(TempStorage& temp_storage, int linear_tid)
      : temp_storage(temp_storage.Alias())
      , linear_tid(linear_tid)
  {}

  //! @}
  //! @name Commutative reductions
  //! @{

  //! @rst
  //! Computes a row-wide reduction and returns the row aggregate to every thread in the row.
  //! The reduction operator must be commutative and associative.
  //!
  //! - @smemreuse
  //! @endrst
  //!
  //! @tparam ReductionOp
  //!   **[inferred]** Commutative binary reduction operator type having member
  //!   ``T operator()(const T &a, const T &b)``.
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @param[in] reduction_op
  //!   Commutative binary reduction operator.
  //!
  //! @param[in] identity
  //!   Identity value for ``reduction_op``.
  //!
  //! @return
  //!   The row-wide reduction aggregate.
  template <typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T CommutativeReduce(T input, ReductionOp reduction_op, T identity)
  {
    const int lane_id = linear_tid % warp_threads;
    if constexpr (WarpsPerRow == 1)
    {
      return WarpAllReduce(input, reduction_op, lane_id);
    }
    else
    {
      const int warp_id      = linear_tid / warp_threads;
      const int row_id       = warp_id / WarpsPerRow;
      const int row_warp_id  = warp_id % WarpsPerRow;
      const T warp_aggregate = WarpReduceT{temp_storage.warp_reduce[warp_id]}.Reduce(input, reduction_op);

      if (lane_id == 0)
      {
        detail::uninitialized_copy_single(&temp_storage.partials[row_id][row_warp_id].Alias(), warp_aggregate);
      }
      __syncthreads();

      T partial = identity;
      if (lane_id < WarpsPerRow)
      {
        partial = temp_storage.partials[row_id][lane_id].Alias();
      }
      const T result = WarpAllReduce(partial, reduction_op, lane_id);
      return result;
    }
  }

  //! @}
  //! @name Sum reductions
  //! @{

  //! @rst
  //! Computes a row-wide sum and returns the row aggregate to every thread in the row.
  //! This overload requires ``cuda::identity_element`` to provide an additive identity for ``T``.
  //!
  //! - @smemreuse
  //! @endrst
  //!
  //! @param[in] input
  //!   Calling thread's input item.
  //!
  //! @return
  //!   The row-wide sum.
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T Sum(T input)
  {
    static_assert(::cuda::has_identity_element_v<::cuda::std::plus<>, T>,
                  "T must have a cuda::identity_element for cuda::std::plus<>; use CommutativeReduce with an explicit "
                  "identity otherwise");
    return CommutativeReduce(input, ::cuda::std::plus<>{}, ::cuda::identity_element<::cuda::std::plus<>, T>());
  }

  //! @}
};

CUB_NAMESPACE_END
