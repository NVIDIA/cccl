//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___OPEN_ADDRESSING_KERNELS_CUH
#define _CUDAX___CUCO___OPEN_ADDRESSING_KERNELS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_reduce.cuh>

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/__detail/utils.hpp>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>
#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::__open_addressing
{
/**
 * @brief Inserts all elements in the range `[first, first + n)` and returns the number of
 * successful insertions if `pred` of the corresponding stencil returns true.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _AtomicT, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __insert_if_n(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _AtomicT* __num_successes,
  _Ref __ref)
{
  using __block_reduce = cub::BlockReduce<typename _Ref::size_type, _BlockSize>;
  __shared__ typename __block_reduce::TempStorage __temp_storage;
  typename _Ref::size_type __thread_num_successes = 0;

  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if (__pred(*(__stencil + __idx)))
    {
      typename ::cuda::std::iterator_traits<_InputIt>::value_type const __insert_element{*(__first + __idx)};
      if constexpr (_CgSize == 1)
      {
        if (__ref.insert(__insert_element))
        {
          __thread_num_successes++;
        }
      }
      else
      {
        const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
          ::cooperative_groups::this_thread_block());
        if (__ref.insert(__tile, __insert_element) && __tile.thread_rank() == 0)
        {
          __thread_num_successes++;
        }
      }
    }
    __idx += __loop_stride;
  }

  const auto __block_num_successes = __block_reduce(__temp_storage).Sum(__thread_num_successes);
  if (threadIdx.x == 0)
  {
    __num_successes->fetch_add(__block_num_successes, ::cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Inserts all elements in the range `[first, first + n)` if `pred` of the corresponding
 * stencil returns true.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __insert_if_n(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if (__pred(*(__stencil + __idx)))
    {
      typename ::cuda::std::iterator_traits<_InputIt>::value_type const __insert_element{*(__first + __idx)};
      if constexpr (_CgSize == 1)
      {
        __ref.insert(__insert_element);
      }
      else
      {
        const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
          ::cooperative_groups::this_thread_block());
        __ref.insert(__tile, __insert_element);
      }
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Erases keys in the range `[first, first + n)` and counts successes.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _AtomicT, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__erase(_InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _AtomicT* __num_successes, _Ref __ref)
{
  using __block_reduce = cub::BlockReduce<typename _Ref::size_type, _BlockSize>;
  __shared__ typename __block_reduce::TempStorage __temp_storage;
  typename _Ref::size_type __thread_num_successes = 0;

  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if constexpr (_CgSize == 1)
    {
      if (__ref.erase(*(__first + __idx)))
      {
        __thread_num_successes++;
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      if (__ref.erase(__tile, *(__first + __idx)) && __tile.thread_rank() == 0)
      {
        __thread_num_successes++;
      }
    }
    __idx += __loop_stride;
  }

  const auto __block_num_successes = __block_reduce(__temp_storage).Sum(__thread_num_successes);
  if (threadIdx.x == 0)
  {
    __num_successes->fetch_add(__block_num_successes, ::cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Erases keys in the range `[first, first + n)` (fire-and-forget, no counting).
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__erase(_InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if constexpr (_CgSize == 1)
    {
      __ref.erase(*(__first + __idx));
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      __ref.erase(__tile, *(__first + __idx));
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Fills a slot array with a given value.
 */

template <int _BlockSize, class _Value>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__fill(_Value* __slots, ::cuda::experimental::cuco::__detail::__index_type __n, _Value __sentinel)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride();
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id();

  while (__idx < __n)
  {
    __slots[__idx] = __sentinel;
    __idx += __loop_stride;
  }
}

/**
 * @brief Applies a callback to all values in the range `[first, first + n)`.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _CallbackOp, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __for_each_n(
  _InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _CallbackOp __callback_op, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if constexpr (_CgSize == 1)
    {
      __ref.for_each(*(__first + __idx), __callback_op);
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      __ref.for_each(__tile, *(__first + __idx), __callback_op);
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Contains test with predicate.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __contains_if_n(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if (__pred(*(__stencil + __idx)))
    {
      bool __found = false;
      if constexpr (_CgSize == 1)
      {
        __found = __ref.contains(*(__first + __idx));
      }
      else
      {
        const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
          ::cooperative_groups::this_thread_block());
        __found = __ref.contains(__tile, *(__first + __idx));
      }
      *(__output_begin + __idx) = __found;
    }
    else
    {
      *(__output_begin + __idx) = false;
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Finds keys in range.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __find_if_n(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if (__pred(*(__stencil + __idx)))
    {
      if constexpr (_CgSize == 1)
      {
        *(__output_begin + __idx) = __ref.find(*(__first + __idx));
      }
      else
      {
        const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
          ::cooperative_groups::this_thread_block());
        *(__output_begin + __idx) = __ref.find(__tile, *(__first + __idx));
      }
    }
    else
    {
      *(__output_begin + __idx) = __ref.end();
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Insert and find, outputs to separate found_begin and inserted_begin arrays.
 */

template <int _CgSize, int _BlockSize, class _InputIt, class _FoundIt, class _InsertedIt, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __insert_and_find(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _FoundIt __found_begin,
  _InsertedIt __inserted_begin,
  _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if constexpr (_CgSize == 1)
    {
      auto const [__iter, __inserted] = __ref.insert_and_find(*(__first + __idx));
      *(__found_begin + __idx)        = __iter;
      *(__inserted_begin + __idx)     = __inserted;
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      auto const [__iter, __inserted] = __ref.insert_and_find(__tile, *(__first + __idx));
      if (__tile.thread_rank() == 0)
      {
        *(__found_begin + __idx)    = __iter;
        *(__inserted_begin + __idx) = __inserted;
      }
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Total count of matches for keys in range using block reduce + atomic counter.
 */

template <bool _IsOuter, int _CgSize, int _BlockSize, class _InputIt, class _AtomicT, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__count(_InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _AtomicT* __counter, _Ref __ref)
{
  using __block_reduce = cub::BlockReduce<typename _Ref::size_type, _BlockSize>;
  __shared__ typename __block_reduce::TempStorage __temp_storage;
  typename _Ref::size_type __thread_count = 0;

  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if constexpr (_CgSize == 1)
    {
      if constexpr (_IsOuter)
      {
        __thread_count += ::cuda::std::max(typename _Ref::size_type{1}, __ref.count(*(__first + __idx)));
      }
      else
      {
        __thread_count += __ref.count(*(__first + __idx));
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      auto __count_val = __ref.count(__tile, *(__first + __idx));
      if (__tile.thread_rank() == 0)
      {
        if constexpr (_IsOuter)
        {
          __thread_count += ::cuda::std::max(typename _Ref::size_type{1}, __count_val);
        }
        else
        {
          __thread_count += __count_val;
        }
      }
    }
    __idx += __loop_stride;
  }

  const auto __block_count = __block_reduce(__temp_storage).Sum(__thread_count);
  if (threadIdx.x == 0)
  {
    __counter->fetch_add(__block_count, ::cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Per-key count of matches (no stencil).
 */

template <bool _IsOuter, int _CgSize, int _BlockSize, class _InputIt, class _OutputIt, class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __count_each(
  _InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _OutputIt __output_begin, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    if constexpr (_CgSize == 1)
    {
      auto __count_val          = __ref.count(*(__first + __idx));
      *(__output_begin + __idx) = _IsOuter ? ::cuda::std::max(decltype(__count_val){1}, __count_val) : __count_val;
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      auto __count_val = __ref.count(__tile, *(__first + __idx));
      if (__tile.thread_rank() == 0)
      {
        *(__output_begin + __idx) = _IsOuter ? ::cuda::std::max(decltype(__count_val){1}, __count_val) : __count_val;
      }
    }
    __idx += __loop_stride;
  }
}

/**
 * @brief Retrieve matching slots.
 */

template <bool _IsOuter,
          int _BlockSize,
          class _InputProbeIt,
          class _StencilIt,
          class _Predicate,
          class _OutputProbeIt,
          class _OutputMatchIt,
          class _AtomicCounter,
          class _Ref>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __retrieve(
  _InputProbeIt __input_probe,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputProbeIt __output_probe,
  _OutputMatchIt __output_match,
  _AtomicCounter __counter,
  _Ref __ref)
{
  const auto __block = ::cooperative_groups::this_thread_block();
  __ref.template retrieve<_IsOuter, _BlockSize>(
    __block, __input_probe, __n, __stencil, __pred, __output_probe, __output_match, __counter);
}

/**
 * @brief Size kernel.
 */

template <int _BlockSize, class _StorageRef, class _OutputIt, class _IsFilled>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __size(_StorageRef __storage, _IsFilled __is_filled, _OutputIt __output)
{
  using __block_reduce = cub::BlockReduce<typename _StorageRef::__size_type, _BlockSize>;
  __shared__ typename __block_reduce::TempStorage __temp_storage;

  typename _StorageRef::__size_type __thread_count = 0;
  const auto __n                                   = __storage.capacity();
  const auto __loop_stride                         = ::cuda::experimental::cuco::__detail::__grid_stride();
  auto __idx                                       = ::cuda::experimental::cuco::__detail::__global_thread_id();

  while (__idx < __n)
  {
    __thread_count += static_cast<typename _StorageRef::__size_type>(__is_filled(*(__storage.data() + __idx)));
    __idx += __loop_stride;
  }

  const auto __block_count = __block_reduce(__temp_storage).Sum(__thread_count);
  if (threadIdx.x == 0)
  {
    __output->fetch_add(__block_count, ::cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief Rehash kernel.
 */

template <int _BlockSize, class _StorageRef, class _ContainerRef, class _Predicate>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__rehash(_StorageRef __storage_ref, _ContainerRef __container_ref, _Predicate __is_filled)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride();
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id();

  while (__idx < __storage_ref.num_buckets())
  {
    const auto __bucket = __storage_ref[__idx];
    for (int __i = 0; __i < _StorageRef::__bucket_size; ++__i)
    {
      const auto __slot = __bucket[__i];
      if (__is_filled(__slot))
      {
        __container_ref.insert(__slot);
      }
    }
    __idx += __loop_stride;
  }
}
} // namespace cuda::experimental::cuco::__open_addressing

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_KERNELS_CUH
