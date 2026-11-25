// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! block_load.cuh Operations for reading linear tiles of data into the CUDA thread block.

#pragma once

#include <cub/config.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/detail/cub_vector.cuh>
#include <cub/detail/it_traits.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/numeric_traits.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/detail/unit_word.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/__memory/__vectorized_storage.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/__type_traits/is_vector_type.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__memory/is_sufficiently_aligned.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/__new/device_new.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/has_unique_object_representation.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <nv/target>

#if __cpp_lib_format >= 201907L
#  include <cuda/std/__concepts/same_as.h>
#  include <cuda/std/__fwd/format.h> // IWYU pragma: keep
#endif // __cpp_lib_format >= 201907L

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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
LoadDirectBlocked(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
{
  // Load directly in thread-blocked order
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = block_src_it[(linear_tid * ItemsPerThread) + i]; // NOLINT(bugprone-misplaced-widening-cast)
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void LoadDirectBlocked(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const auto src_pos = (linear_tid * ItemsPerThread) + i;
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void LoadDirectBlocked(
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

  _CUB::LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
//! @brief Whether the storage of ``RandomAccessIterator`` can be loaded with vectors
// TODO(bgruber): remove the check for is_primitive<T> in CCCL 4.0
template <typename RandomAccessIterator, typename T>
inline constexpr bool block_load_can_vectorize_v =
  THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIterator>
  && (::cuda::is_trivially_copyable_v<T> || is_primitive_v<T>)
  && ::cuda::std::is_same_v<it_value_t<RandomAccessIterator>, T>
  // handles contiguous iterators wrapping volatile pointers, like span<volatile int>::iterator
  && !::cuda::std::is_volatile_v<::cuda::std::remove_reference_t<it_reference_t<RandomAccessIterator>>>;

//! @brief The width in bytes of the widest vectors with which ``LoadDirectBlockedVectorized`` loads vectors of
//! ``VecSize`` consecutive items of ``InputT``, wider than 16 bytes only from sm_100
// A vector holds at most VecSize items, and is only wider than 16 bytes without a cache modifier or with LOAD_LDG, the
// only cache modifier with which ThreadLoad loads 32 byte words. ThreadLoad copies volatile vectors in 8 byte words,
// thus wider volatile vectors only add the tests of their alignment
template <typename InputT, int VecSize, CacheLoadModifier Modifier>
inline constexpr ::cuda::std::size_t block_load_max_vector_access_v = ::cuda::std::min(
  ::cuda::std::size_t{1} << ::cuda::std::countr_zero(VecSize * sizeof(InputT)),
  Modifier == LOAD_DEFAULT || Modifier == LOAD_LDG ? ::cuda::__max_memory_access_width
  : Modifier == LOAD_VOLATILE
    ? ::cuda::std::size_t{8}
    : ::cuda::std::size_t{16});

//! @brief Whether every byte of ``T`` holds its value: arithmetic and extended floating point types, CUDA vector types,
//! and types with unique object representations
template <typename T>
inline constexpr bool block_load_is_padding_free_v =
  ::cuda::std::__is_extended_arithmetic_v<T> || ::cuda::is_vector_type_v<T>
  || ::cuda::std::has_unique_object_representations_v<T>;

//! @brief Whether the vectors of ``VecSize`` items of ``InputT`` are copied into the items by every vector width rather
//! than once all widths join
// Joined, the words of padded items spanning several words narrow the vector loads to the members of the items
template <typename InputT, int VecSize>
inline constexpr bool block_load_copy_per_width_v =
  !block_load_is_padding_free_v<InputT>
  && sizeof(::cuda::__vectorized_storage_word_t<InputT, VecSize>) < sizeof(InputT);

//! @brief Copies the items of ``InputT`` in ``vectors`` into ``dst_items``: as words, or one by one if they are
//! converted to ``T`` or copied by every vector width
template <typename InputT, typename VectorT, int NumVectors, typename T, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
copy_vector_items(const VectorT (&vectors)[NumVectors], T (&dst_items)[ItemsPerThread])
{
  constexpr int vec_size = ItemsPerThread / NumVectors;
  using unit_t _CCCL_NODEBUG =
    ::cuda::std::conditional_t<::cuda::std::is_same_v<T, InputT> && !block_load_copy_per_width_v<InputT, vec_size>,
                               ::cuda::__vectorized_storage_word_t<InputT, vec_size>,
                               InputT>;
  constexpr int units_per_vector = int{sizeof(VectorT) / sizeof(unit_t)};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumVectors; i++)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int j = 0; j < units_per_vector; j++)
    {
      const unit_t unit = vectors[i].template __alias<unit_t>(j);
      // dst_items may be aligned to less than a word, thus each word is copied with memcpy from a copy of it: a memcpy
      // straight from the vectors is split into bytes
      if constexpr (::cuda::std::is_same_v<unit_t, InputT>)
      {
        dst_items[(i * units_per_vector) + j] = static_cast<T>(unit);
      }
      else
      {
        ::cuda::std::memcpy(
          reinterpret_cast<unsigned char*>(dst_items) + (((i * units_per_vector) + j) * int{sizeof(unit_t)}),
          &unit,
          sizeof(unit_t));
      }
    }
  }
}

//! @brief Loads the ``NumVectors`` vectors of ``VecSize`` items of a thread with accesses of ``Alignment`` bytes, which
//! ``src_ptr`` must be aligned to, ``VectorStride`` vectors apart from vector ``vector_index`` after ``src_ptr`` on
template <::cuda::std::size_t Alignment,
          int VectorStride,
          int VecSize,
          CacheLoadModifier Modifier,
          typename InputT,
          typename VectorT,
          int NumVectors>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load_direct_vectors(const InputT* src_ptr, unsigned int vector_index, VectorT (&vectors)[NumVectors])
{
  using aligned_vector_t _CCCL_NODEBUG = ::cuda::__as_vectorized_storage_t<InputT, VecSize, Alignment>;
  const aligned_vector_t* vec_ptr =
    reinterpret_cast<const aligned_vector_t*>(::cuda::std::__runtime_assume_aligned(src_ptr, Alignment)) + vector_index;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumVectors; i++)
  {
    vectors[i] = _CUB::ThreadLoad<Modifier>(vec_ptr + (i * VectorStride));
  }
}

//! @brief Loads the vectors of ``VecSize`` items of a thread in the arrangement of ``LoadDirectStripedVectorized`` item
//! by item, starting with the vector at ``thread_src_it``
template <int ThreadsPerBlock, int VecSize, typename RandomAccessIterator, typename T, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load_direct_striped_vectors(RandomAccessIterator thread_src_it, T (&dst_items)[ItemsPerThread])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread / VecSize; i++)
  {
    _CUB::LoadDirectBlocked(i * ThreadsPerBlock, thread_src_it, reinterpret_cast<T(&)[VecSize]>(dst_items[i * VecSize]));
  }
}

//! @brief Loads the ``ItemsPerThread / VecSize`` vectors of ``VecSize`` items of a thread, ``VectorStride`` vectors
//! apart from vector ``vector_index`` after ``src_ptr`` on, with the widest accesses that the alignment of ``src_ptr``
//! allows, or item by item from ``thread_src_ptr``, the first item of the thread
template <int VectorStride, int VecSize, CacheLoadModifier Modifier, typename InputT, typename T, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void load_direct_vectorized(
  const InputT* src_ptr, unsigned int vector_index, const InputT* thread_src_ptr, T (&dst_items)[ItemsPerThread])
{
  constexpr auto max_vector_access = block_load_max_vector_access_v<InputT, VecSize, Modifier>;
  if constexpr (block_load_can_vectorize_v<const InputT*, InputT> && 4 <= max_vector_access
                && alignof(InputT) < max_vector_access)
  {
    constexpr int vectors_per_thread = ItemsPerThread / VecSize;
    using vector_t _CCCL_NODEBUG     = ::cuda::__as_vectorized_storage_t<InputT, VecSize, max_vector_access>;

    // Every width only loads the vectors, and the fallback stores its items into them, so that all of them join before
    // a single copy into dst_items. Every vector starts a whole number of vectors after src_ptr, thus a single test
    // selects the accesses of all vectors of the thread. The vectorized loads must be inlined: a call prevents the
    // compiler from merging them with the fallback. Only sm_100 and newer load vectors wider than 16 bytes with a
    // single access
    [[maybe_unused]] vector_t vectors[vectors_per_thread];
    while (true)
    {
      if constexpr (alignof(InputT) < 32 && 32 <= max_vector_access)
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_100, ({
            if (::cuda::std::is_sufficiently_aligned<32>(src_ptr))
            {
              _CUB::detail::load_direct_vectors<32, VectorStride, VecSize, Modifier>(src_ptr, vector_index, vectors);
              if constexpr (block_load_copy_per_width_v<InputT, VecSize>)
              { // Padded items spanning several words are copied by every width instead of after the join
                _CUB::detail::copy_vector_items<InputT>(vectors, dst_items);
              }
              break;
            }
          }))
      }
      if constexpr (alignof(InputT) < 16 && 16 <= max_vector_access)
      {
        if (::cuda::std::is_sufficiently_aligned<16>(src_ptr))
        {
          _CUB::detail::load_direct_vectors<16, VectorStride, VecSize, Modifier>(src_ptr, vector_index, vectors);
          if constexpr (block_load_copy_per_width_v<InputT, VecSize>)
          { // Padded items spanning several words are copied by every width instead of after the join
            _CUB::detail::copy_vector_items<InputT>(vectors, dst_items);
          }
          break;
        }
      }
      if constexpr (alignof(InputT) < 8 && 8 <= max_vector_access)
      {
        if (::cuda::std::is_sufficiently_aligned<8>(src_ptr))
        {
          _CUB::detail::load_direct_vectors<8, VectorStride, VecSize, Modifier>(src_ptr, vector_index, vectors);
          if constexpr (block_load_copy_per_width_v<InputT, VecSize>)
          { // Padded items spanning several words are copied by every width instead of after the join
            _CUB::detail::copy_vector_items<InputT>(vectors, dst_items);
          }
          break;
        }
      }
      if constexpr (alignof(InputT) < 4 && 4 <= max_vector_access)
      {
        if (::cuda::std::is_sufficiently_aligned<4>(src_ptr))
        {
          _CUB::detail::load_direct_vectors<4, VectorStride, VecSize, Modifier>(src_ptr, vector_index, vectors);
          if constexpr (block_load_copy_per_width_v<InputT, VecSize>)
          { // Padded items spanning several words are copied by every width instead of after the join
            _CUB::detail::copy_vector_items<InputT>(vectors, dst_items);
          }
          break;
        }
      }
      // Items without padding are stored into the vectors to join the widths, padded items go straight into dst_items,
      // because packing their members into words costs more than the join saves
      if constexpr (block_load_is_padding_free_v<InputT>)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < vectors_per_thread; i++)
        {
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int j = 0; j < VecSize; j++)
          {
            vectors[i].template __alias<InputT>(j) = thread_src_ptr[(i * VectorStride * VecSize) + j];
          }
        }
        break;
      }
      else
      {
        _CUB::detail::load_direct_striped_vectors<VectorStride, VecSize>(thread_src_ptr, dst_items);
        return;
      }
    }

    // Padded items spanning several words are copied by every width before the join. Otherwise, their vector
    // loads narrow to loads of their members
    if constexpr (!block_load_copy_per_width_v<InputT, VecSize>)
    {
      _CUB::detail::copy_vector_items<InputT>(vectors, dst_items);
    }
  }
  else
  {
    _CUB::detail::load_direct_striped_vectors<VectorStride, VecSize>(thread_src_ptr, dst_items);
  }
}

//! @brief The widest ``CubVector`` of up to four ``UnitWord`` device words that divides ``VecSize`` consecutive items
//! of ``InputT``, whose items each vector of ``BLOCK_LOAD_STRIPED_VECTORIZED`` holds
template <typename InputT, int VecSize>
struct block_load_vector
{
  // Find biggest memory access word that InputT is a whole multiple of
  using device_word_t              = typename UnitWord<InputT>::DeviceWord;
  static constexpr int total_words = static_cast<int>(VecSize * sizeof(InputT) / sizeof(device_word_t));
  static constexpr int vector_size = (total_words % 4 == 0) ? 4 : (total_words % 2 == 0) ? 2 : 1;

  using type = typename CubVector<device_word_t, vector_size>::Type;
};

template <typename InputT, int VecSize>
using block_load_vector_t = typename block_load_vector<InputT, VecSize>::type;
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

//! @rst
//! Load a linear segment of items into a blocked arrangement across the thread block.
//!
//! .. versionadded:: 2.2.0
//!    First appears in CUDA Toolkit 12.3.
//!
//! @blocked
//!
//! Each thread loads its items with the widest vectors that the alignment of ``block_src_ptr`` allows, up to 16 bytes
//! wide. On sm_100 and newer, loads without a cache modifier or with ``LOAD_LDG`` use vectors up to 32 bytes wide.
//!
//! The following conditions will prevent vectorization and loading will fall back to cub::BLOCK_LOAD_DIRECT without the
//! cache modifier:
//!
//! - ``block_src_ptr`` or the items of a thread do not allow vectors of at least 4 bytes that are wider than the
//!   alignment of ``InputT``
//! - ``InputT`` is volatile or not trivially copyable
//!
//! The fallback also drops ``LOAD_VOLATILE``, thus its loads are not volatile.
//!
//! @endrst
//!
//! @tparam Modifier
//!   **[optional]** Cache load modifier for reading the input (default: ``LOAD_DEFAULT``)
//!
//! @tparam VecSize
//!   **[optional]** The maximum number of consecutive items in a vector, which must evenly divide ``ItemsPerThread``.
//!   ``0`` is the same as ``ItemsPerThread`` (default: ``0``)
//!
//! @tparam T
//!   **[inferred]** The data type to load.
//!
//! @tparam ItemsPerThread
//!   **[inferred]** The number of consecutive items partitioned onto each thread.
//!
//! @tparam InputT
//!   **[inferred]** The value type of the input, which is loaded with vectors and converted to ``T``.
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
template <CacheLoadModifier Modifier = LOAD_DEFAULT, int VecSize = 0, typename T, int ItemsPerThread, typename InputT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
LoadDirectBlockedVectorized(int linear_tid, const InputT* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  constexpr int vec_size = VecSize == 0 ? ItemsPerThread : VecSize;
  static_assert(ItemsPerThread % vec_size == 0, "VecSize must evenly divide ItemsPerThread");

  // The blocked arrangement is the striped arrangement of a block of a single thread, whose vectors are consecutive
  // from vector linear_tid * (ItemsPerThread / vec_size) on. The offset of the vectors is unsigned, because a signed
  // one is sign-extended before the tests, while the first item of the thread is offset in signed arithmetic like in
  // LoadDirectBlocked, because an unsigned offset spills DeviceTopK on sm_90
  _CUB::detail::load_direct_vectorized<1, vec_size, Modifier>(
    block_src_ptr,
    static_cast<unsigned int>(linear_tid) * (ItemsPerThread / vec_size),
    block_src_ptr + (linear_tid * ItemsPerThread),
    dst_items);
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
// Accepts explicitly specified T and ItemsPerThread, which cannot bind to the leading Modifier of the overload above
template <typename T, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
LoadDirectBlockedVectorized(int linear_tid, T* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  _CUB::LoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, block_src_ptr, dst_items);
}

template <CacheLoadModifier MODIFIER, typename T, int ItemsPerThread>
CCCL_DEPRECATED_BECAUSE("Use LoadDirectBlockedVectorized instead") _CCCL_DEVICE_API _CCCL_FORCEINLINE void
InternalLoadDirectBlockedVectorized(int linear_tid, const T* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  _CUB::LoadDirectBlockedVectorized<MODIFIER>(linear_tid, block_src_ptr, dst_items);
}
#endif // _CCCL_DOXYGEN_INVOKED

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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
LoadDirectStriped(int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = block_src_it[linear_tid + (i * ThreadsPerBlock)]; // NOLINT(bugprone-misplaced-widening-cast)
  }
}

namespace detail
{
//! @brief Loads the arrangement of ``LoadDirectStripedVectorized`` with vectors of type
//! ``CubVector<InputT, min(ItemsPerThread, VecSize)>::Type``, to which ``block_src_ptr`` must be aligned
template <int ThreadsPerBlock,
          ::cuda::std::size_t VecSize,
          CacheLoadModifier Modifier,
          typename InputT,
          typename T,
          int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
load_direct_striped_vectorized_aligned(int linear_tid, const InputT* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  constexpr int vec_size = (::cuda::std::min) (ItemsPerThread, static_cast<int>(VecSize));
  static_assert(ItemsPerThread % vec_size == 0, "The vector size must evenly divide ItemsPerThread");

  using vector_t                   = typename CubVector<InputT, vec_size>::Type;
  constexpr int vectors_per_thread = ItemsPerThread / vec_size;

  // Vector i of a thread is vector linear_tid + i * ThreadsPerBlock of the block
  // The thread offset is computed in unsigned arithmetic like offsets from threadIdx.x, because a signed multiplication
  // generates additional address arithmetic in the calling kernels
  const vector_t* vec_ptr =
    reinterpret_cast<const vector_t*>(block_src_ptr + (static_cast<unsigned int>(linear_tid) * vec_size));

  // Raw storage, because the vector of a type without default constructor has none either
  alignas(vector_t) unsigned char vec_bytes[vectors_per_thread * sizeof(vector_t)];
  vector_t* vec_items = reinterpret_cast<vector_t*>(vec_bytes);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < vectors_per_thread; i++)
  {
    vec_items[i] = _CUB::ThreadLoad<Modifier>(vec_ptr + (i * ThreadsPerBlock));
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] = reinterpret_cast<const InputT*>(vec_bytes)[i];
  }
}
} // namespace detail

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
//! Each vector is loaded as a ``CubVector<InputT, VecSize>::Type``. The following conditions will prevent
//! vectorization and the same arrangement is loaded item by item:
//!
//! - ``block_src_ptr`` is not aligned to the vector access width
//! - ``InputT`` is volatile or not trivially copyable
//!
//! @endrst
//!
//! @tparam ThreadsPerBlock
//!   The thread block size in threads
//!
//! @tparam VecSize
//!   The number of consecutive items in each vector, from 1 to 4. Must evenly divide ``ItemsPerThread``.
//!
//! @tparam Modifier
//!   **[optional]** Cache load modifier for reading the input (default: ``LOAD_DEFAULT``)
//!
//! @tparam InputT
//!   **[inferred]** The value type of the input, which is loaded with vectors and converted to ``T``.
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
//! @param[in] block_src_ptr
//!   The thread block's base pointer for loading from
//!
//! @param[out] dst_items
//!   Destination to load data into
template <int ThreadsPerBlock,
          ::cuda::std::size_t VecSize,
          CacheLoadModifier Modifier = LOAD_DEFAULT,
          typename InputT,
          typename T,
          int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
LoadDirectStripedVectorized(int linear_tid, const InputT* block_src_ptr, T (&dst_items)[ItemsPerThread])
{
  static_assert(ItemsPerThread % VecSize == 0, "VecSize must evenly divide ItemsPerThread");
  constexpr int vec_size = static_cast<int>(VecSize);

  // TODO(bgruber): remove the check for is_primitive<InputT> in CCCL 4.0
  constexpr bool can_vectorize =
    !::cuda::std::is_volatile_v<InputT> && (::cuda::is_trivially_copyable_v<InputT> || detail::is_primitive_v<InputT>);
  if constexpr (can_vectorize)
  {
    if (::cuda::std::is_sufficiently_aligned<alignof(typename CubVector<InputT, vec_size>::Type)>(block_src_ptr))
    {
      _CUB::detail::load_direct_striped_vectorized_aligned<ThreadsPerBlock, VecSize, Modifier>(
        linear_tid, block_src_ptr, dst_items);
      return;
    }
  }

  // Load the same arrangement item by item
  const InputT* thread_src_ptr = block_src_ptr + (static_cast<unsigned int>(linear_tid) * vec_size);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread / vec_size; i++)
  {
    if constexpr (Modifier != LOAD_DEFAULT)
    {
      _CUB::LoadDirectBlocked(i * ThreadsPerBlock,
                              CacheModifiedInputIterator<Modifier, InputT, int>(thread_src_ptr),
                              reinterpret_cast<T(&)[vec_size]>(dst_items[i * vec_size]));
    }
    else
    {
      _CUB::LoadDirectBlocked(
        i * ThreadsPerBlock, thread_src_ptr, reinterpret_cast<T(&)[vec_size]>(dst_items[i * vec_size]));
    }
  }
}

namespace detail
{
template <int ThreadsPerBlock, typename T, int ItemsPerThread, typename RandomAccessIterator, typename TransformOpT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void load_transform_direct_striped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], TransformOpT transform_op)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    dst_items[i] =
      transform_op(block_src_it[linear_tid + (i * ThreadsPerBlock)]); // NOLINT(bugprone-misplaced-widening-cast)
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void LoadDirectStriped(
  int linear_tid, RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < ItemsPerThread; i++)
  {
    const auto src_pos = linear_tid + (i * ThreadsPerBlock);
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void LoadDirectStriped(
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

  _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end);
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void LoadDirectWarpStriped(
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
_CCCL_DEVICE_API _CCCL_FORCEINLINE void LoadDirectWarpStriped(
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

  _CUB::LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
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
  //!   - The ``RandomAccessIterator`` is not a contiguous iterator (``cub::CacheModifiedInputIterator`` is an exception
  //!     and is still vectorized)
  //!   - The value type of the ``RandomAccessIterator`` is not ``T``
  //!   - The block input offset is not aligned to more than ``T`` and to at least 4 bytes, otherwise the widest vectors
  //!     that its alignment allows are used
  //!   - The input is volatile or ``T`` is not trivially copyable
  //!
  //! - Inputs that are not vectorized are loaded without the cache modifier of a cub::CacheModifiedInputIterator.
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
  //! ``t + ThreadsPerBlock``, and so on. Each vector holds as many items as fit into four words as wide as the
  //! alignment of ``T``, at most 16 bytes, but at least one, as far as the items of a thread allow, and is loaded as a
  //! ``CubVector`` of these items. Thus each vector holds four items of an arithmetic type if the items of a thread are
  //! a multiple of four, and a single item of a type like ``char3`` or ``float3``, which is the arrangement of
  //! cub::BLOCK_LOAD_STRIPED.
  //!
  //! Performance Considerations
  //! ++++++++++++++++++++++++++
  //!
  //! - Consecutive threads load consecutive vectors, so the accesses stay coalesced regardless of the number of items
  //!   per thread.
  //! - The following conditions will prevent vectorization and the same arrangement is loaded item by item:
  //!
  //!   - The ``RandomAccessIterator`` is not a contiguous iterator (``cub::CacheModifiedInputIterator`` is an exception
  //!     and is still vectorized)
  //!   - The value type of the ``RandomAccessIterator`` is not ``T``
  //!   - The block input offset is not aligned to the vector access width
  //!   - The input is volatile or ``T`` is not trivially copyable
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

_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, BlockLoadAlgorithm algo)
{
  return os << _CUB::detail::to_string(algo);
}
#endif // _CCCL_HOSTED() && !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END

#if __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)
template <::cuda::std::same_as<char> CharT>
struct std::formatter<_CUB::BlockLoadAlgorithm, CharT> : formatter<const CharT*, CharT>
{
  template <class FmtCtx>
  _CCCL_HOST_API auto format(const _CUB::BlockLoadAlgorithm& algo, FmtCtx& ctx) const
  {
    return formatter<const CharT*, CharT>::format(_CUB::detail::to_string(algo), ctx);
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

  using _TempStorage _CCCL_NODEBUG = decltype(temp_storage_helper());

  // Internal storage allocator
  _CCCL_DEVICE_API _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockLoad()
      : temp_storage(PrivateStorage())
      , linear_tid(_CUB::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @brief Collective constructor using the specified memory allocation as temporary storage.
  //!
  //! @rst
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //! @endrst
  //!
  //! @param[in] temp_storage Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockLoad(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , linear_tid(_CUB::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Load(RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread])
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT)
    {
      _CUB::LoadDirectBlocked(linear_tid, block_src_it, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED)
    {
      _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_VECTORIZE)
    {
      if constexpr (detail::is_CacheModifiedInputIterator<RandomAccessIterator>
                    && ::cuda::std::is_same_v<detail::it_value_t<RandomAccessIterator>, T>)
      {
        _CUB::LoadDirectBlockedVectorized<RandomAccessIterator::__modifier>(linear_tid, block_src_it.ptr, dst_items);
      }
      else if constexpr (detail::block_load_can_vectorize_v<RandomAccessIterator, T>)
      {
        _CUB::LoadDirectBlockedVectorized<LOAD_DEFAULT>(linear_tid, ::cuda::std::to_address(block_src_it), dst_items);
      }
      else
      {
        _CUB::LoadDirectBlocked(linear_tid, block_src_it, dst_items);
      }
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      // Vectors of up to four device words of T, or single items if they do not hold a whole number of T
      using vector_t         = detail::block_load_vector_t<T, ItemsPerThread>;
      constexpr int vec_size = sizeof(vector_t) % sizeof(T) == 0 ? static_cast<int>(sizeof(vector_t) / sizeof(T)) : 1;
      if constexpr (detail::is_CacheModifiedInputIterator<RandomAccessIterator>
                    && ::cuda::std::is_same_v<detail::it_value_t<RandomAccessIterator>, T>)
      {
        _CUB::LoadDirectStripedVectorized<ThreadsPerBlock, vec_size, RandomAccessIterator::__modifier>(
          linear_tid, block_src_it.ptr, dst_items);
      }
      else if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIterator>
                         && ::cuda::std::is_same_v<detail::it_value_t<RandomAccessIterator>, T>)
      {
        _CUB::LoadDirectStripedVectorized<ThreadsPerBlock, vec_size, LOAD_DEFAULT>(
          linear_tid, ::cuda::std::to_address(block_src_it), dst_items);
      }
      else
      {
        // Load the same arrangement item by item
        const auto thread_src_it = block_src_it + (static_cast<unsigned int>(linear_tid) * vec_size);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < ItemsPerThread / vec_size; i++)
        {
          _CUB::LoadDirectBlocked(
            i * ThreadsPerBlock, thread_src_it, reinterpret_cast<T(&)[vec_size]>(dst_items[i * vec_size]));
        }
      }
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE)
    {
      _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items);
      block_exchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_WARP_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      _CUB::LoadDirectWarpStriped(linear_tid, block_src_it, dst_items);
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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Load(RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end)
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT || Algorithm == BLOCK_LOAD_VECTORIZE)
    {
      _CUB::LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED || Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE)
    {
      _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end);
      block_exchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_WARP_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      _CUB::LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end);
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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  Load(RandomAccessIterator block_src_it, T (&dst_items)[ItemsPerThread], int block_items_end, DefaultT oob_default)
  {
    if constexpr (Algorithm == BLOCK_LOAD_DIRECT || Algorithm == BLOCK_LOAD_VECTORIZE)
    {
      _CUB::LoadDirectBlocked(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_STRIPED || Algorithm == BLOCK_LOAD_STRIPED_VECTORIZED)
    {
      _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_TRANSPOSE)
    {
      _CUB::LoadDirectStriped<ThreadsPerBlock>(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      block_exchange(temp_storage).StripedToBlocked(dst_items, dst_items);
    }
    else if constexpr (Algorithm == BLOCK_LOAD_WARP_TRANSPOSE || Algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED)
    {
      _CUB::LoadDirectWarpStriped(linear_tid, block_src_it, dst_items, block_items_end, oob_default);
      block_exchange(temp_storage).WarpStripedToBlocked(dst_items, dst_items);
    }
  }

  //! @}
};

template <class Policy, class It, class T = cub::detail::it_value_t<It>>
struct BlockLoadType
{
  using type _CCCL_NODEBUG = cub::BlockLoad<T, Policy::BLOCK_THREADS, Policy::ITEMS_PER_THREAD, Policy::LOAD_ALGORITHM>;
};

CUB_NAMESPACE_END
