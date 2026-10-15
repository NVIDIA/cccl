// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! block_load.cuh Operations for reading linear tiles of data into the CUDA thread block.

#pragma once

#include <cub/config.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__memory/as_bytes_vectorized.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__fwd/format.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__memory/is_sufficiently_aligned.h>
#include <cuda/std/__new/device_new.h>

CUB_NAMESPACE_BEGIN

//! @name Blocked arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base input iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <typename T, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlocked(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
{
  // Load directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = block_src_it[linear_tid * ItemsPerThread + i]; // NOLINT(bugprone-misplaced-widening-cast)
  }
}

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block, guarded by range.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   First out-of-bounds index when loading from block_src_it
template <typename T, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectBlocked(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const auto src_pos = linear_tid * ItemsPerThread + i;
    if (src_pos < block_items_end)
    {
      dst_items[i] = block_src_it[src_pos];
    }
  }
}

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block, guarded
//! by range, with a fall-back assignment of out-of-bound elements.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//!
//! @endrst
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base input iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   First out-of-bounds index when loading from block_src_it
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <typename T, typename DefaultT, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectBlocked(
  int linear_tid,
  RandomAccessIterator block_src_it,
  T (&dst_items)[ItemsPerThread],
  int block_items_end,
  DefaultT oob_default)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = oob_default;
  }

  LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

//! @brief Internal implementation of LoadDirectBlockedVectorized for input aligned to the vector type
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_ptr
//!   Input pointer for loading from, aligned to ``cuda::__as_bytes_vectorized_t<T, ItemsPerThread>``
//!
//! @param[out] dst_items
//!   Destination to load data into
template <CacheLoadModifier MODIFIER, typename T, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void
InternalLoadDirectBlockedVectorized(int linear_tid, const T* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  // All items of a thread are loaded with the widest vectors that evenly divide them
  using vector_t = ::cuda::__as_bytes_vectorized_t<T, ItemsPerThread>;
  _CCCL_ASSERT(::cuda::std::is_sufficiently_aligned<alignof(vector_t)>(block_src_ptr),
               "block_src_ptr must be aligned to the vector type");

  // Load in thread-blocked order
  const vector_t vec_items = ThreadLoad<MODIFIER>(reinterpret_cast<const vector_t*>(block_src_ptr) + linear_tid);

  // Copy to destination
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = vec_items.template __alias<T>()[i];
  }
}

#endif // _CCCL_DOXYGEN_INVOKED

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//!
//! The input offset (``block_ptr + block_offset``) must be quad-item aligned
//!
//! The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
//!
//! - ``ItemsPerThread`` is odd
//! - The data type ``T`` is not a built-in primitive or CUDA vector type
//!   (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
//!
//! @endrst
//!
//! @tparam Modifier
//!   **[optional]** Cache load modifier for the vectorized loads (default: ``LOAD_DEFAULT``)
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//!   linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_ptr
//!   The thread block's base pointer for loading from
//!
//! @param[out] dst_items
//!  destination to load data into
template <CacheLoadModifier Modifier = LOAD_DEFAULT, typename T, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectBlockedVectorized(int linear_tid, const T* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  if constexpr (!::cuda::__is_vectorizable<T>())
  {
    LoadDirectBlocked(linear_tid, block_src_ptr, dst_items);
  }
  else
  {
    using vector_t = ::cuda::__as_bytes_vectorized_t<T, ItemsPerThread>;
    if constexpr (ItemsPerThread == 1)
    {
      if (::cuda::std::is_sufficiently_aligned<alignof(vector_t)>(block_src_ptr))
      {
        const vector_t vec_items = ThreadLoad<Modifier>(reinterpret_cast<const vector_t*>(block_src_ptr) + linear_tid);
        dst_items[0]             = vec_items.template __alias<T>()[0];
      }
      else
      {
        LoadDirectBlocked(linear_tid, block_src_ptr, dst_items);
      }
    }
    // Add the alignment check to ensure the vectorized loading can proceed.
    else if (::cuda::std::is_sufficiently_aligned<alignof(vector_t)>(block_src_ptr))
    {
      InternalLoadDirectBlockedVectorized<Modifier>(linear_tid, block_src_ptr, dst_items);
    }
    else
    {
      LoadDirectBlocked(linear_tid, block_src_ptr, dst_items);
    }
  }
}

//! @}
//! @name Striped arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam ThreadsPerBlock
//!   The thread block size in threads
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <int ThreadsPerBlock, typename T, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectStriped(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = block_src_it[linear_tid + i * ThreadsPerBlock]; // NOLINT(bugprone-misplaced-widening-cast)
  }
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

//! @brief Internal implementation of LoadDirectStripedVectorized for input aligned to the vector type
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_ptr
//!   Input pointer for loading from, aligned to ``cuda::__as_bytes_vectorized_t<T, VecSize>``
//!
//! @param[out] dst_items
//!   Destination to load data into
template <int ThreadsPerBlock, int VecSize, CacheLoadModifier Modifier, typename T, int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void
InternalLoadDirectStripedVectorized(int linear_tid, const T* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  static_assert(ItemsPerThread % VecSize == 0, "VecSize must evenly divide ItemsPerThread");

  // Vector i of a thread is the blocked vector of thread linear_tid + i * ThreadsPerBlock
  // The thread offset is computed in unsigned arithmetic like offsets from threadIdx.x, because a signed multiplication
  // generates additional address arithmetic in the calling kernels
  const T* thread_src_ptr = block_src_ptr + static_cast<unsigned int>(linear_tid) * VecSize;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread / VecSize; i++)
  {
    InternalLoadDirectBlockedVectorized<Modifier>(
      i * ThreadsPerBlock, thread_src_ptr, reinterpret_cast<T(&)[VecSize]>(dst_items[i * VecSize]));
  }
}

#endif // _CCCL_DOXYGEN_INVOKED

//! @rst
//! Load a linear segment of items into a striped arrangement of vectors across the thread block.
//!
//! .. versionadded:: 3.6.0
//!
//! Assumes a :ref:`striped arrangement <flexible-data-arrangement>` of vectors of ``VecSize`` consecutive items across
//! the thread block, where *thread*\ :sub:`i` owns the vectors (*i*), (*i* + *block-threads*), ...,
//! (*i* + (*block-threads* * (*items-per-thread* / ``VecSize`` - 1))). With ``VecSize == 1``, this is the arrangement
//! of ``LoadDirectStriped``.
//!
//! The following conditions will prevent vectorization and the same arrangement is loaded item by item:
//!
//! - The ``RandomAccessIterator`` is not contiguous
//! - ``block_src_it`` is not aligned to the largest power of two that divides ``sizeof(T) * VecSize``, up to 32 bytes
//!   (16 bytes before CUDA 13)
//! - The data type ``T`` is aligned to more than 32 bytes (16 bytes before CUDA 13)
//!
//! @endrst
//!
//! @tparam ThreadsPerBlock
//!   The thread block size in threads
//!
//! @tparam VecSize
//!   The number of consecutive items in each vector. Must evenly divide ``ItemsPerThread``.
//!
//! @tparam Modifier
//!   **[optional]** Cache load modifier for the vectorized loads (default: ``LOAD_DEFAULT``)
//!
//! @tparam RandomAccessIterator
//!   **[inferred]** The random-access iterator type for input iterator.
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of items partitioned onto each thread.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) + linear_tid` for 2D
//!   thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <int ThreadsPerBlock,
          int VecSize,
          CacheLoadModifier Modifier = LOAD_DEFAULT,
          typename RandomAccessIterator,
          typename T,
          int ItemsPerThread>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectStripedVectorized(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
{
  static_assert(ItemsPerThread % VecSize == 0, "VecSize must evenly divide ItemsPerThread");
  if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIterator> && ::cuda::__is_vectorizable<T>())
  {
    const T* block_src_ptr = ::cuda::std::to_address(block_src_it);
    // Add the alignment check to ensure the vectorized loading can proceed.
    if (::cuda::std::is_sufficiently_aligned<alignof(::cuda::__as_bytes_vectorized_t<T, VecSize>)>(block_src_ptr))
    {
      InternalLoadDirectStripedVectorized<ThreadsPerBlock, VecSize, Modifier>(linear_tid, block_src_ptr, dst_items);
      return;
    }
  }

  // Load the same arrangement item by item
  const auto thread_src_it = block_src_it + static_cast<unsigned int>(linear_tid) * VecSize;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread / VecSize; i++)
  {
    LoadDirectBlocked(i * ThreadsPerBlock, thread_src_it, reinterpret_cast<T(&)[VecSize]>(dst_items[i * VecSize]));
  }
}

namespace detail
{
template <int ThreadsPerBlock, typename T, int ItemsPerThread, typename RandomAccessIterator, typename TransformOpT>
_CCCL_DEVICE _CCCL_FORCEINLINE void load_transform_direct_striped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], TransformOpT transform_op)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] =
      transform_op(block_src_it[linear_tid + i * ThreadsPerBlock]); // NOLINT(bugprone-misplaced-widening-cast)
  }
}
} // namespace detail

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block, guarded by range
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam ThreadsPerBlock
//!   The thread block size in threads
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) +
//! linear_tid</tt> for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
template <int ThreadsPerBlock, typename T, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectStriped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const auto src_pos = linear_tid + i * ThreadsPerBlock;
    if (src_pos < block_items_end)
    {
      dst_items[i] = block_src_it[src_pos];
    }
  }
}

//! @rst
//! Load a linear segment of items into a striped arrangement across the thread block, guarded
//! by range, with a fall-back assignment of out-of-bound elements.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @striped
//!
//! @endrst
//!
//! @tparam ThreadsPerBlock
//!   The thread block size in threads
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <int ThreadsPerBlock, typename T, typename DefaultT, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectStriped(
  int linear_tid,
  RandomAccessIterator block_src_it,
  T (&dst_items)[ItemsPerThread],
  int block_items_end,
  DefaultT oob_default)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = oob_default;
  }

  LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end);
}

//! @}
//! @name Warp-striped arrangement I/O (direct)
//! @{

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @endrst
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <typename T, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void
LoadDirectWarpStriped(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
{
  const int tid         = linear_tid & (detail::warp_threads - 1);
  const int wid         = linear_tid >> detail::log2_warp_threads;
  const int warp_offset = wid * detail::warp_threads * ItemsPerThread;

  // Load directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    new (&dst_items[i])
      T(block_src_it[warp_offset + tid + (i * detail::warp_threads)]); // NOLINT(bugprone-misplaced-widening-cast)
  }
}

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block, guarded by range
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @warpstriped
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @endrst
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
template <typename T, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectWarpStriped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
{
  const int tid         = linear_tid & (detail::warp_threads - 1);
  const int wid         = linear_tid >> detail::log2_warp_threads;
  const int warp_offset = wid * detail::warp_threads * ItemsPerThread;

  // Load directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const auto src_pos = warp_offset + tid + (i * detail::warp_threads);
    if (src_pos < block_items_end)
    {
      new (&dst_items[i]) T(block_src_it[src_pos]);
    }
  }
}

//! @rst
//! Load a linear segment of items into a warp-striped arrangement across the thread block,
//! guarded by range, with a fall-back assignment of out-of-bound elements.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @warpstriped
//!
//! @endrst
//!
//! Usage Considerations
//! ++++++++++++++++++++
//!
//! The number of threads in the thread block must be a multiple of the architecture's warp size.
//!
//! @tparam T
//!   **inferred** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **inferred** The number of consecutive items partitioned onto each thread.
//!
//! @tparam RandomAccessIterator
//!   **inferred** The random-access iterator type for input \iterator.
//!
//! @param[in] linear_tid
//!   A suitable 1D thread-identifier for the calling thread (e.g., `(threadIdx.y * blockDim.x) +
//! linear_tid` for 2D thread blocks)
//!
//! @param[in] block_src_it
//!   The thread block's base iterator for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
//!
//! @param[in] block_items_end
//!   Number of valid items to load
//!
//! @param[in] oob_default
//!   Default value to assign out-of-bound items
template <typename T, typename DefaultT, int ItemsPerThread, typename RandomAccessIterator>
_CCCL_DEVICE _CCCL_FORCEINLINE void LoadDirectWarpStriped(
  int linear_tid,
  RandomAccessIterator block_src_it,
  T (&dst_items)[ItemsPerThread],
  int block_items_end,
  DefaultT oob_default)
{
  // Load directly in warp-striped order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = oob_default;
  }

  LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
}

//! @}

//! @brief cub::BlockLoadAlgorithm enumerates alternative algorithms for cub::BlockLoad to read a linear segment of data
//!        from memory into a blocked arrangement across a CUDA thread block.
enum BlockLoadAlgorithm // NOLINT(cppcoreguidelines-use-enum-class)
{
  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) decreases as the access stride between threads increases
  //! (i.e., the number items per thread).
  //! @endrst
  BLOCK_LOAD_DIRECT,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! The utilization of memory transactions (coalescing) doesn't depend on the number of items per thread.
  //!
  //! @endrst
  BLOCK_LOAD_STRIPED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read from memory using CUDA's built-in
  //! vectorized loads as a coalescing optimization. For example, ``ld.global.v4.s32`` instructions will be generated
  //! when ``T = int`` and ``ItemsPerThread % 4 == 0``.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high until the the access stride between threads
  //!   (i.e., the number items per thread) exceeds the maximum vector load width (typically 4 items or 64B, whichever
  //!   is lower).
  //! - The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT:
  //!
  //!   - ``ItemsPerThread`` is odd
  //!   - The ``RandomAccessIterator`` is not a simple pointer type
  //!   - The block input offset is not quadword-aligned
  //!   - The data type ``T`` is not a built-in primitive or CUDA vector type
  //!     (e.g., ``short``, ``int2``, ``double``, ``float2``, etc.)
  //!
  //! @endrst
  BLOCK_LOAD_VECTORIZE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of data is read efficiently from memory and then locally
  //! transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - The local reordering incurs slightly longer latencies and throughput than the direct cub::BLOCK_LOAD_DIRECT and
  //!   cub::BLOCK_LOAD_VECTORIZE alternatives.
  //!
  //! @endrst
  BLOCK_LOAD_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read efficiently from memory and then
  //! locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - ThreadsPerBlock must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - The local reordering incurs slightly larger latencies than the direct cub::BLOCK_LOAD_DIRECT and
  //!   cub::BLOCK_LOAD_VECTORIZE alternatives.
  //! - Provisions more shared storage, but incurs smaller latencies than the
  //!   BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED alternative.
  //!
  //! @endrst
  BLOCK_LOAD_WARP_TRANSPOSE,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Like ``BLOCK_LOAD_WARP_TRANSPOSE``, a :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read
  //! directly from memory and then is locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
  //! To reduce the shared memory requirement, only one warp's worth of shared memory is provisioned and is subsequently
  //! time-sliced among warps.
  //!
  //! Usage Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - ThreadsPerBlock must be a multiple of WARP_THREADS
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - The utilization of memory transactions (coalescing) remains high regardless of items loaded per thread.
  //! - Provisions less shared memory temporary storage, but incurs larger latencies than the BLOCK_LOAD_WARP_TRANSPOSE
  //!   alternative.
  //!
  //! @endrst
  BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,

  //! @rst
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! A :ref:`striped arrangement <flexible-data-arrangement>` of vectors is read from memory using CUDA's built-in
  //! vectorized loads. Each vector holds consecutive items, and thread ``t`` owns the vectors at positions ``t``,
  //! ``t + ThreadsPerBlock``, and so on. The vectors are as wide as the vector loads of cub::BLOCK_LOAD_VECTORIZE. If
  //! those do not hold a whole number of items, each vector is a single item, which is the arrangement of
  //! cub::BLOCK_LOAD_STRIPED.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - Consecutive threads load consecutive vectors, so the accesses stay coalesced regardless of the number of items
  //!   per thread.
  //! - The following conditions will prevent vectorization and the same arrangement is loaded item by item:
  //!
  //!   - The ``RandomAccessIterator`` is not contiguous
  //!   - The block input offset is not aligned to the vector
  //!   - The data type ``T`` is aligned to more than 32 bytes (16 bytes before CUDA 13)
  //!
  //! - Guarded loads use the arrangement of cub::BLOCK_LOAD_STRIPED.
  //!
  //! @endrst
  BLOCK_LOAD_STRIPED_VECTORIZED,
};

#if _CCCL_HOSTED() && !defined(_CCCL_DOXYGEN_INVOKED)
namespace detail
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(BlockLoadAlgorithm algo) noexcept
{
  switch (algo)
  {
    case BLOCK_LOAD_DIRECT:
      return "BLOCK_LOAD_DIRECT";
    case BLOCK_LOAD_STRIPED:
      return "BLOCK_LOAD_STRIPED";
    case BLOCK_LOAD_VECTORIZE:
      return "BLOCK_LOAD_VECTORIZE";
    case BLOCK_LOAD_TRANSPOSE:
      return "BLOCK_LOAD_TRANSPOSE";
    case BLOCK_LOAD_WARP_TRANSPOSE:
      return "BLOCK_LOAD_WARP_TRANSPOSE";
    case BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED:
      return "BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED";
    case BLOCK_LOAD_STRIPED_VECTORIZED:
      return "BLOCK_LOAD_STRIPED_VECTORIZED";
  }
  return "<unknown BlockLoadAlgorithm>";
}
} // namespace detail

inline ::std::ostream& operator<<(::std::ostream& os, BlockLoadAlgorithm algo)
{
  return os << CUB_NS_QUALIFIER::detail::to_string(algo);
}
#endif // _CCCL_HOSTED() && !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END

#if __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)
template <::cuda::std::same_as<char> CharT>
struct std::formatter<CUB_NS_QUALIFIER::BlockLoadAlgorithm, CharT> : formatter<const CharT*, CharT>
{
  template <class FmtCtx>
  auto format(const CUB_NS_QUALIFIER::BlockLoadAlgorithm& algo, FmtCtx& ctx) const
  {
    return formatter<const CharT*, CharT>::format(CUB_NS_QUALIFIER::detail::to_string(algo), ctx);
  }
};
#endif // __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)

CUB_NAMESPACE_BEGIN

//! @rst
//! The BlockLoad class provides :ref:`collective <collective-primitives>` data movement methods for loading a linear
//! segment of items from memory into a :ref:`blocked arrangement <flexible-data-arrangement>` across a CUDA thread
//! block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - The BlockLoad class provides a single data movement abstraction that can be specialized to implement different
//!   cub::BlockLoadAlgorithm strategies. This facilitates different performance policies for different architectures,
//!   data types, granularity sizes, etc.
//! - BlockLoad can be optionally specialized by different data movement strategies:
//!
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_DIRECT`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_STRIPED`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_VECTORIZE`:
//!      A :ref:`blocked arrangement <flexible-data-arrangement>` of data is read directly from memory
//!      using CUDA's built-in vectorized loads as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_STRIPED_VECTORIZED`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of vectors is read directly from memory
//!      using CUDA's built-in vectorized loads as a coalescing optimization.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_TRANSPOSE`:
//!      A :ref:`striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_WARP_TRANSPOSE`:
//!      A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>`.
//!   #. :cpp:enumerator:`cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED`:
//!      A :ref:`warp-striped arrangement <flexible-data-arrangement>` of data is read directly from memory and is then
//!      locally transposed into a :ref:`blocked arrangement <flexible-data-arrangement>` one warp at a time.
//!
//! - @rowmajor
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockLoad}
//!
//! The code snippet below illustrates the loading of a linear segment of 512 integers into a "blocked" arrangement
//! across 128 threads where each thread owns 4 consecutive items. The load is specialized for
//! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
//! pattern (after which items are locally reordered among threads).
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
//!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
//!
//!        // Allocate shared memory for BlockLoad
//!        __shared__ typename BlockLoad::TempStorage temp_storage;
//!
//!        // Load a segment of consecutive items that are blocked across threads
//!        int thread_data[4];
//!        BlockLoad(temp_storage).Load(d_data, thread_data);
//!    }
//!
//! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``. The set of ``thread_data`` across the block of threads in
//! those threads will be ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of dynamically shared memory with
//! BlockReduce and how to re-purpose the same memory region. This example can be easily adapted to the storage required
//! by BlockLoad.
//!
//! @endrst
//!
//! @tparam T
//!   The data type to read into (which must be convertible from the input iterator's value type).
//!
//! @tparam BlockDimX
//!   The thread block length in threads along the X dimension
//!
//! @tparam ItemsPerThread
//!   The number of consecutive items partitioned onto each thread.
//!
//! @tparam Algorithm
//!   **[optional]** cub::BlockLoadAlgorithm tuning policy. default: ``cub::BLOCK_LOAD_DIRECT``.
//!
//! @tparam BlockDimY
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BlockDimZ
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
template <typename T,
          int BlockDimX,
          int ItemsPerThread,
          BlockLoadAlgorithm Algorithm = BLOCK_LOAD_DIRECT,
          int BlockDimY                = 1,
          int BlockDimZ                = 1>
class BlockLoad
{
  static constexpr int ThreadsPerBlock = BlockDimX * BlockDimY * BlockDimZ; // total threads in the block

  // transposing load algorithms need a BlockExchange
  using block_exchange =
    BlockExchange<T,
                  BlockDimX,
                  ItemsPerThread,
                  /* WarpTimeSlicing = */ Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
                  BlockDimY,
                  BlockDimZ>;

  static_assert((Algorithm != BLOCK_LOAD_WARP_TRANSPOSE && Algorithm != BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
                  || (ThreadsPerBlock % detail::warp_threads == 0),
                "ThreadsPerBlock must be a multiple of warp_threads for this BlockLoadAlgorithm");

  _CCCL_HOST_DEVICE_API static constexpr auto temp_storage_helper()
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT || Algorithm == BLOCK_LOAD_STRIPED || Algorithm == BLOCK_LOAD_VECTORIZE
                  || Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      return NullType{};
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE
                       || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      return typename block_exchange::TempStorage{};
    }
  }

  using _TempStorage = decltype(temp_storage_helper());

  // Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  _TempStorage& temp_storage;
  int linear_tid;

public:
  /// @smemstorage{BlockLoad}
  using TempStorage = Uninitialized<_TempStorage>;

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  //!
  //! @rst
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //! @endrst
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoad()
      : temp_storage(PrivateStorage())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @brief Collective constructor using the specified memory allocation as temporary storage.
  //!
  //! @rst
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoad(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @}
  //! @name Data movement
  //! @{

  //! @rst
  //! Load a linear segment of items from memory.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the loading of a linear segment of 512 integers into a "blocked" arrangement
  //! across 128 threads where each thread owns 4 consecutive items. The load is specialized for
  //! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data);
  //!    }
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, ...``. The set of ``thread_data`` across the block of threads
  //! in those threads will be ``{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }``.
  //!
  //! @endrst
  //!
  //! @param[in] block_src_it
  //!   The thread block's base iterator for loading from
  //!
  //! @param[out] dst_items
  //!   Destination to load data into
  template <typename RandomAccessIterator>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED)
    {
      LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_VECTORIZE)
    {
      if constexpr (detail::is_CacheModifiedInputIterator<RandomAccessIterator>)
      {
        LoadDirectBlockedVectorized<RandomAccessIterator::__modifier>(linear_tid, block_src_it.ptr, dst_items);
      }
      else if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIterator>)
      {
        LoadDirectBlockedVectorized(linear_tid, ::cuda::std::to_address(block_src_it), dst_items);
      }
      else
      {
        LoadDirectBlocked(linear_tid, block_src_it, dst_items);
      }
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      // Vectors of the width BLOCK_LOAD_VECTORIZE loads, or single items if that does not hold a whole number of T
      using vector_word_t = typename ::cuda::__as_bytes_vectorized_t<T, ItemsPerThread>::value_type;
      constexpr int vec_size =
        sizeof(vector_word_t) % sizeof(T) == 0 ? static_cast<int>(sizeof(vector_word_t) / sizeof(T)) : 1;
      if constexpr (detail::is_CacheModifiedInputIterator<RandomAccessIterator>)
      {
        LoadDirectStripedVectorized<ThreadsPerBlock, vec_size, RandomAccessIterator::__modifier>(
          linear_tid, block_src_it.ptr, dst_items);
      }
      else
      {
        LoadDirectStripedVectorized<ThreadsPerBlock, vec_size>(linear_tid, block_src_it, dst_items);
      }
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE)
    {
      LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items);
      block_exchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_WARP_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items);
      block_exchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }
  }

  //! @rst
  //! Load a linear segment of items from memory, guarded by range.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded loading of a linear segment of 512 integers into a "blocked"
  //! arrangement across 128 threads where each thread owns 4 consecutive items. The load is specialized for
  //! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int block_items_end, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data, block_items_end);
  //!    }
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, 6...`` and ``block_items_end`` is ``5``. The set of
  //! ``thread_data`` across the block of threads in those threads will be ``{ [0,1,2,3], [4,?,?,?], ..., [?,?,?,?] }``,
  //! with only the first two threads being unmasked to load portions of valid data (and other items remaining
  //! unassigned).
  //!
  //! @endrst
  //!
  //! @param[in] block_src_it
  //!   The thread block's base iterator for loading from
  //!
  //! @param[out] dst_items
  //!   Destination to load data into
  //!
  //! @param[in] block_items_end
  //!   Number of valid items to load
  template <typename RandomAccessIterator>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Load(RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT || Algorithm == BLOCK_LOAD_VECTORIZE)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED || Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE)
    {
      LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end);
      block_exchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_WARP_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
      block_exchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }
  }

  //! @rst
  //! Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - @blocked
  //! - @smemreuse
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates the guarded loading of a linear segment of 512 integers into a "blocked"
  //! arrangement across 128 threads where each thread owns 4 consecutive items. The load is specialized for
  //! ``BLOCK_LOAD_WARP_TRANSPOSE``, meaning memory references are efficiently coalesced using a warp-striped access
  //! pattern (after which items are locally reordered among threads).
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_load.cuh>
  //!
  //!    __global__ void ExampleKernel(int *d_data, int block_items_end, ...)
  //!    {
  //!        // Specialize BlockLoad for a 1D block of 128 threads owning 4 integer items each
  //!        using BlockLoad = cub::BlockLoad<int, 128, 4, BLOCK_LOAD_WARP_TRANSPOSE>;
  //!
  //!        // Allocate shared memory for BlockLoad
  //!        __shared__ typename BlockLoad::TempStorage temp_storage;
  //!
  //!        // Load a segment of consecutive items that are blocked across threads
  //!        int thread_data[4];
  //!        BlockLoad(temp_storage).Load(d_data, thread_data, block_items_end, -1);
  //!    }
  //!
  //! Suppose the input ``d_data`` is ``0, 1, 2, 3, 4, 5, 6...``, ``block_items_end`` is ``5``, and the out-of-bounds
  //! default is ``-1``. The set of ``thread_data`` across the block of threads in those threads will be
  //! ``{ [0,1,2,3], [4,-1,-1,-1], ..., [-1,-1,-1,-1] }``, with only the first two threads being unmasked to load
  //! portions of valid data (and other items are assigned ``-1``)
  //!
  //! @endrst
  //!
  //! @param[in] block_src_it
  //!   The thread block's base iterator for loading from
  //!
  //! @param[out] dst_items
  //!   Destination to load data into
  //!
  //! @param[in] block_items_end
  //!   Number of valid items to load
  //!
  //! @param[in] oob_default
  //!   Default value to assign out-of-bound items
  template <typename RandomAccessIterator, typename DefaultT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Load(RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end, DefaultT oob_default)
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT || Algorithm == BLOCK_LOAD_VECTORIZE)
    {
      LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED || Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE)
    {
      LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      block_exchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_WARP_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      block_exchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }
  }

  //! @}
};

template <class Policy, class It, class T = cub::detail::it_value_t<It>>
struct BlockLoadType
{
  using type = cub::BlockLoad<T, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, Policy::LOAD_ALGORITHM>;
};

CUB_NAMESPACE_END
