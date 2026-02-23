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
//! A Simple Example
//! ++++++++++++++++
//!
//! @blockcollective{BlockShuffle}
//!
//! The code snippet below illustrates shifting a :ref:`blocked arrangement <flexible-data-arrangement>`
//! of 512 integer items up by one item across 128 threads where each thread owns
//! 4 consecutive items.
//!
//! .. tab-set-code::
//!
//!    .. code-block:: c++
//!
//!        #include <cub/cub.cuh>   // or equivalently <cub/block/block_shuffle.cuh>
//!
//!        __global__ void ExampleKernel(int* d_data, ...)
//!        {
//!            // Specialize BlockShuffle for a 1D block of 128 threads owning 4 integer items each
//!            using BlockShuffle = cub::BlockShuffle<int, 128>;
//!
//!            // Allocate shared memory for BlockShuffle
//!            __shared__ typename BlockShuffle::TempStorage temp_storage;
//!
//!            int block_offset = blockIdx.x * 128 * 4;
//!
//!            // Obtain a segment of consecutive items that are blocked across threads
//!            int thread_data[4];
//!            cub::LoadDirectBlocked(threadIdx.x, d_data + block_offset, thread_data);
//!
//!            // Collectively shift blocked items up by one position
//!            BlockShuffle(temp_storage).Up(thread_data, thread_data);
//!
//!            // Store the shifted segment
//!            cub::StoreDirectBlocked(threadIdx.x, d_data + block_offset, thread_data);
//!        }
//!
//!    .. code-block:: python
//!
//!        from numba import cuda
//!        from cuda import coop
//!
//!        items_per_thread = 4
//!
//!        @cuda.jit
//!        def kernel(d_data):
//!            temp_storage = coop.TempStorage()
//!            thread_data = coop.ThreadData(items_per_thread)
//!
//!            block_offset = cuda.blockIdx.x * cuda.blockDim.x * items_per_thread
//!            coop.block.load(d_data[block_offset:], thread_data)
//!
//!            coop.block.shuffle[temp_storage](
//!                thread_data,
//!                thread_data,
//!                block_shuffle_type=coop.block.BlockShuffleType.Up,
//!            )
//!
//!            coop.block.store(d_data[block_offset:], thread_data)
//!
//!        # Launch with one block of 128 threads.
//!        # kernel[1, 128](d_data)
//!
//! Suppose the set of input ``thread_data`` across the block of threads is
//! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }``.
//! The corresponding output ``thread_data`` in those threads will be
//! ``{ [0,0,1,2], [3,4,5,6], [7,8,9,10], ..., [507,508,509,510] }``.
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
  //!
  //! @rst
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //! @endrst
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockShuffle()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  /**
   * @brief Collective constructor using the specified memory allocation
   *        as temporary storage.
   *
   * @rst
   * .. versionadded:: 2.2.0
   *    First appears in CUDA Toolkit 12.3.
   * @endrst
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockShuffle(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @}
  //! @name Shuffle movement
  //! @{

  //! @rst
  //! Each *thread*\ :sub:`i` obtains the ``input`` provided by *thread*\ :sub:`i + distance`.
  //! The offset ``distance`` may be negative.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @smemreuse
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
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @smemreuse
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
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
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
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
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
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
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
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @granularity
  //! - @smemreuse
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

  //! @}
};

CUB_NAMESPACE_END
