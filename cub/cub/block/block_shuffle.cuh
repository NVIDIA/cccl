// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! The cub::BlockShuffle class provides :ref:`collective <collective-primitives>` methods for shuffling data
//! partitioned across a CUDA thread block.

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

CUB_NAMESPACE_BEGIN

//! @rst
//! The BlockShuffle class provides :ref:`collective <collective-primitives>`
//! methods for shuffling data partitioned across a CUDA thread block.
//!
//! Overview
//! ++++++++++++++++
//!
//! It is commonplace for blocks of threads to rearrange data items between threads.
//! The BlockShuffle abstraction allows threads to efficiently shift items either
//! (a) up to their successor or
//! (b) down to their predecessor
//!
//! @endrst
//!
//! @tparam T
//!   The data type to be exchanged.
//!
//! @tparam BlockDimX
//!   The thread block length in threads along the X dimension
//!
//! @tparam BlockDimY
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BlockDimZ
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T, int BlockDimX, int BlockDimY = 1, int BlockDimZ = 1>
class BlockShuffle
{
private:
  static constexpr int BLOCK_THREADS = BlockDimX * BlockDimY * BlockDimZ;

  static constexpr int LOG_WARP_THREADS = detail::log2_warp_threads;
  static constexpr int WARP_THREADS     = 1 << LOG_WARP_THREADS;
  static constexpr int WARPS            = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;

  /// Shared memory storage layout type (last element from each thread's input)
  using _TempStorage = T[BLOCK_THREADS];

public:
  /// \smemstorage{BlockShuffle}
  struct TempStorage : Uninitialized<_TempStorage>
  {};

private:
  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  unsigned int linear_tid;

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

public:
  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockShuffle()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation
   *        as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockShuffle(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @}  end member group
  //! @name Shuffle movement
  //! @{

  //! @rst
  //!
  //! Each *thread*\ :sub:`i` obtains the ``input`` provided by *thread*\ :sub:`i + distance`.
  //! The offset ``distance`` may be negative.
  //!
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   @rst
  //!   The input item from the calling thread (*thread*\ :sub:`i`)
  //!   @endrst
  //!
  //! @param[out] output
  //!   @rst
  //!   The ``input`` item from the successor (or predecessor) thread
  //!   *thread*\ :sub:`i + distance` (may be aliased to ``input``).
  //!   This value is only updated for for *thread*\ :sub:`i` when
  //!   ``0 <= (i + distance) < BLOCK_THREADS - 1``
  //!   @endrst
  //!
  //! @param[in] distance
  //!   Offset distance (may be negative)
  _CCCL_DEVICE _CCCL_FORCEINLINE void Offset(T input, T& output, int distance = 1)
  {
    temp_storage[linear_tid] = input;

    __syncthreads();

    const int offset_tid = static_cast<int>(linear_tid) + distance;
    if ((offset_tid >= 0) && (offset_tid < BLOCK_THREADS))
    {
      output = temp_storage[static_cast<size_t>(offset_tid)];
    }
  }

  //! @rst
  //! Each *thread*\ :sub:`i` obtains the ``input`` provided by *thread*\ :sub:`i + distance`.
  //!
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input item
  //!
  //! @param[out] output
  //!   @rst
  //!   The ``input`` item from thread
  //!   *thread*\ :sub:`(i + distance>) % BLOCK_THREADS` (may be aliased to ``input``).
  //!   This value is not updated for *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!   @endrst
  //!
  //! @param[in] distance
  //!   Offset distance (`0 < distance < `BLOCK_THREADS`)
  _CCCL_DEVICE _CCCL_FORCEINLINE void Rotate(T input, T& output, unsigned int distance = 1)
  {
    temp_storage[linear_tid] = input;

    __syncthreads();

    unsigned int offset = linear_tid + distance;
    if (offset >= BLOCK_THREADS)
    {
      offset -= BLOCK_THREADS;
    }

    output = temp_storage[offset];
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>` of
  //! ``input`` items, shifting it up by one item.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The item ``prev[0]`` is not updated for *thread*\ :sub:`0`.
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Up(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD])
  {
    temp_storage[linear_tid] = input[ITEMS_PER_THREAD - 1];

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = ITEMS_PER_THREAD - 1; ITEM > 0; --ITEM)
    {
      prev[ITEM] = input[ITEM - 1];
    }

    if (linear_tid > 0)
    {
      prev[0] = temp_storage[linear_tid - 1];
    }
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>`
  //! of ``input`` items, shifting it up by one item. All threads receive the ``input`` provided by
  //! *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The item ``prev[0]`` is not updated for *thread*\ :sub:`0`.
  //!   @endrst
  //!
  //! @param[out] block_suffix
  //!   @rst
  //!   The item ``input[ITEMS_PER_THREAD - 1]`` from *thread*\ :sub:`BLOCK_THREADS - 1`, provided to all threads
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Up(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD], T& block_suffix)
  {
    Up(input, prev);
    block_suffix = temp_storage[BLOCK_THREADS - 1];
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>`
  //! of ``input`` items, shifting it down by one item.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The value ``prev[0]`` is not updated for *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Down(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD])
  {
    temp_storage[linear_tid] = input[0];

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD - 1; ITEM++)
    {
      prev[ITEM] = input[ITEM + 1];
    }

    if (linear_tid < BLOCK_THREADS - 1)
    {
      prev[ITEMS_PER_THREAD - 1] = temp_storage[linear_tid + 1];
    }
  }

  //! @rst
  //! The thread block rotates its :ref:`blocked arrangement <flexible-data-arrangement>` of input items,
  //! shifting it down by one item. All threads receive ``input[0]`` provided by *thread*\ :sub:`0`.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
  //!
  //! @endrst
  //!
  //! @param[in] input
  //!   The calling thread's input items
  //!
  //! @param[out] prev
  //!   @rst
  //!   The corresponding predecessor items (may be aliased to ``input``).
  //!   The value ``prev[0]`` is not updated for *thread*\ :sub:`BLOCK_THREADS - 1`.
  //!   @endrst
  //!
  //! @param[out] block_prefix
  //!   @rst
  //!   The item ``input[0]`` from *thread*\ :sub:`0`, provided to all threads
  //!   @endrst
  template <int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Down(T (&input)[ITEMS_PER_THREAD], T (&prev)[ITEMS_PER_THREAD], T& block_prefix)
  {
    Down(input, prev);
    block_prefix = temp_storage[0];
  }

  //! @} end member group
};

CUB_NAMESPACE_END
