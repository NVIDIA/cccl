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
#include <cuda/std/iterator>

#include <cuda/experimental/__cuco/__detail/utils.hpp>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>
#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Inserts all elements in the range `[first, first + n)` and returns the number of
//! successful insertions if `pred` of the corresponding stencil returns true.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _AtomicT, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __insert_if_n(
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

//! @brief Inserts all elements in the range `[first, first + n)` if `pred` of the corresponding
//! stencil returns true.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __insert_if_n(
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

//! @brief Erases keys in the range `[first, first + n)` (fire-and-forget, no counting).
template <int _CgSize, int _BlockSize, class _InputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void
__erase(_InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    typename ::cuda::std::iterator_traits<_InputIt>::value_type const __erase_element{*(__first + __idx)};
    if constexpr (_CgSize == 1)
    {
      __ref.erase(__erase_element);
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      __ref.erase(__tile, __erase_element);
    }
    __idx += __loop_stride;
  }
}

//! @brief Fills a slot array with a given value.
template <int _BlockSize, class _Value>
_CCCL_KERNEL_ATTRIBUTES void
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

//! @brief Applies a callback to all values in the range `[first, first + n)`.
template <int _CgSize, int _BlockSize, class _InputIt, class _CallbackOp, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __for_each_n(
  _InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _CallbackOp __callback_op, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  while (__idx < __n)
  {
    typename ::cuda::std::iterator_traits<_InputIt>::value_type const __key{*(__first + __idx)};
    if constexpr (_CgSize == 1)
    {
      __ref.for_each(__key, __callback_op);
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      __ref.for_each(__tile, __key, __callback_op);
    }
    __idx += __loop_stride;
  }
}

//! @brief Contains test with predicate.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __contains_if_n(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __ref)
{
  const auto __block       = ::cooperative_groups::this_thread_block();
  const auto __thread_idx  = __block.thread_rank();
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  __shared__ bool __output_buffer[_BlockSize / _CgSize];

  while ((__idx - __thread_idx / _CgSize) < __n) // the whole thread block falls into the same iteration
  {
    if constexpr (_CgSize == 1)
    {
      if (__idx < __n)
      {
        typename ::cuda::std::iterator_traits<_InputIt>::value_type const __key = *(__first + __idx);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        __output_buffer[__thread_idx] = __pred(*(__stencil + __idx)) ? __ref.contains(__key) : false;
      }
      __block.sync();
      if (__idx < __n)
      {
        *(__output_begin + __idx) = __output_buffer[__thread_idx];
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(__block);
      if (__idx < __n)
      {
        typename ::cuda::std::iterator_traits<_InputIt>::value_type const __key = *(__first + __idx);
        const auto __found = __pred(*(__stencil + __idx)) ? __ref.contains(__tile, __key) : false;
        if (__tile.thread_rank() == 0)
        {
          *(__output_begin + __idx) = __found;
        }
      }
    }
    __idx += __loop_stride;
  }
}

//! @brief Finds keys in range.
template <int _CgSize, int _BlockSize, class _InputIt, class _StencilIt, class _Predicate, class _OutputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __find_if_n(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _StencilIt __stencil,
  _Predicate __pred,
  _OutputIt __output_begin,
  _Ref __ref)
{
  const auto __block       = ::cooperative_groups::this_thread_block();
  const auto __thread_idx  = __block.thread_rank();
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  using __output_type = typename _Ref::iterator;
  __shared__ __output_type __output_buffer[_BlockSize / _CgSize];

  while ((__idx - __thread_idx / _CgSize) < __n) // the whole thread block falls into the same iteration
  {
    if constexpr (_CgSize == 1)
    {
      if (__idx < __n)
      {
        typename ::cuda::std::iterator_traits<_InputIt>::value_type const __key = *(__first + __idx);
        const auto __found                                                      = __ref.find(__key);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        __output_buffer[__thread_idx] = __pred(*(__stencil + __idx)) ? __found : __ref.end();
      }
      __block.sync();
      if (__idx < __n)
      {
        *(__output_begin + __idx) = __output_buffer[__thread_idx];
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(__block);
      if (__idx < __n)
      {
        typename ::cuda::std::iterator_traits<_InputIt>::value_type const __key = *(__first + __idx);
        const auto __found                                                      = __ref.find(__tile, __key);
        if (__tile.thread_rank() == 0)
        {
          *(__output_begin + __idx) = __pred(*(__stencil + __idx)) ? __found : __ref.end();
        }
      }
    }
    __idx += __loop_stride;
  }
}

//! @brief Insert and find, outputs to separate found_begin and inserted_begin arrays.
template <int _CgSize, int _BlockSize, class _InputIt, class _FoundIt, class _InsertedIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __insert_and_find(
  _InputIt __first,
  ::cuda::experimental::cuco::__detail::__index_type __n,
  _FoundIt __found_begin,
  _InsertedIt __inserted_begin,
  _Ref __ref)
{
  const auto __block       = ::cooperative_groups::this_thread_block();
  const auto __thread_idx  = __block.thread_rank();
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  using __output_type = typename _Ref::iterator;
  __shared__ __output_type __output_found_buffer[_BlockSize / _CgSize];
  __shared__ bool __output_inserted_buffer[_BlockSize / _CgSize];

  while ((__idx - __thread_idx / _CgSize) < __n) // the whole thread block falls into the same iteration
  {
    if constexpr (_CgSize == 1)
    {
      if (__idx < __n)
      {
        typename ::cuda::std::iterator_traits<_InputIt>::value_type const __insert_element{*(__first + __idx)};
        const auto [__iter, __inserted] = __ref.insert_and_find(__insert_element);
        /*
         * The ld.relaxed.gpu instruction causes L1 to flush more frequently, causing increased
         * sector stores from L2 to global memory. By writing results to shared memory and then
         * synchronizing before writing back to global, we no longer rely on L1, preventing the
         * increase in sector stores from L2 to global and improving performance.
         */
        __output_found_buffer[__thread_idx]    = __iter;
        __output_inserted_buffer[__thread_idx] = __inserted;
      }
      __block.sync();
      if (__idx < __n)
      {
        *(__found_begin + __idx)    = __output_found_buffer[__thread_idx];
        *(__inserted_begin + __idx) = __output_inserted_buffer[__thread_idx];
      }
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(__block);
      if (__idx < __n)
      {
        typename ::cuda::std::iterator_traits<_InputIt>::value_type const __insert_element{*(__first + __idx)};
        const auto [__iter, __inserted] = __ref.insert_and_find(__tile, __insert_element);
        if (__tile.thread_rank() == 0)
        {
          *(__found_begin + __idx)    = __iter;
          *(__inserted_begin + __idx) = __inserted;
        }
      }
    }
    __idx += __loop_stride;
  }
}

//! @brief Total count of matches for keys in range using block reduce + atomic counter.
template <int _CgSize, int _BlockSize, class _InputIt, class _AtomicT, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void
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
      __thread_count += __ref.count(*(__first + __idx));
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      const auto __tile_count = ::cooperative_groups::reduce(
        __tile, __ref.count(__tile, *(__first + __idx)), ::cooperative_groups::plus<typename _Ref::size_type>());
      if (__tile.thread_rank() == 0)
      {
        __thread_count += __tile_count;
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

//! @brief Per-key count of matches (no stencil).
template <int _CgSize, int _BlockSize, class _InputIt, class _OutputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __count_each(
  _InputIt __first, ::cuda::experimental::cuco::__detail::__index_type __n, _OutputIt __output_begin, _Ref __ref)
{
  const auto __loop_stride = ::cuda::experimental::cuco::__detail::__grid_stride() / _CgSize;
  auto __idx               = ::cuda::experimental::cuco::__detail::__global_thread_id() / _CgSize;

  using __size_type = typename _Ref::size_type;

  while (__idx < __n)
  {
    typename ::cuda::std::iterator_traits<_InputIt>::value_type const __key = *(__first + __idx);
    if constexpr (_CgSize == 1)
    {
      *(__output_begin + __idx) = __ref.count(__key);
    }
    else
    {
      const auto __tile = ::cooperative_groups::tiled_partition<_CgSize, ::cooperative_groups::thread_block>(
        ::cooperative_groups::this_thread_block());
      const auto __total =
        ::cooperative_groups::reduce(__tile, __ref.count(__tile, __key), ::cooperative_groups::plus<__size_type>());
      if (__tile.thread_rank() == 0)
      {
        *(__output_begin + __idx) = __total;
      }
    }
    __idx += __loop_stride;
  }
}

//! @brief Retrieve matching slots.
template <int _BlockSize,
          class _InputProbeIt,
          class _StencilIt,
          class _Predicate,
          class _OutputProbeIt,
          class _OutputMatchIt,
          class _AtomicCounter,
          class _Ref>
_CCCL_KERNEL_ATTRIBUTES void __retrieve(
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
  __ref.template retrieve<_BlockSize>(
    __block, __input_probe, __n, __stencil, __pred, __output_probe, __output_match, __counter);
}

//! @brief Size kernel.
template <int _BlockSize, class _StorageRef, class _OutputIt, class _IsFilled>
_CCCL_KERNEL_ATTRIBUTES void __size(_StorageRef __storage, _IsFilled __is_filled, _OutputIt __output)
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

//! @brief Rehash kernel.
template <int _BlockSize, class _StorageRef, class _ContainerRef, class _Predicate>
_CCCL_KERNEL_ATTRIBUTES void __rehash(_StorageRef __storage_ref, _ContainerRef __container_ref, _Predicate __is_filled)
{
  __shared__ typename _ContainerRef::value_type __buffer[_BlockSize];
  __shared__ unsigned int __buffer_size;

  static constexpr auto __cg_size = _ContainerRef::cg_size;
  const auto __block              = ::cooperative_groups::this_thread_block();
  const auto __tile = ::cooperative_groups::tiled_partition<__cg_size, ::cooperative_groups::thread_block>(__block);

  const auto __thread_rank                = __block.thread_rank();
  static constexpr auto __tiles_per_block = _BlockSize / __cg_size;
  const auto __tile_rank                  = __tile.meta_group_rank();
  const auto __loop_stride                = ::cuda::experimental::cuco::__detail::__grid_stride();
  auto __idx                              = ::cuda::experimental::cuco::__detail::__global_thread_id();
  const auto __n                          = __storage_ref.num_buckets();

  while (__idx - __thread_rank < __n)
  {
    if (__thread_rank == 0)
    {
      __buffer_size = 0;
    }
    __block.sync();

    // Gather filled slots from the old storage into shared memory
    if (__idx < __n)
    {
      const auto __bucket = __storage_ref[__idx];
      for (auto const& __slot : __bucket)
      {
        if (__is_filled(__slot))
        {
          __buffer[atomicAdd_block(&__buffer_size, 1u)] = __slot;
        }
      }
    }
    __block.sync();

    const auto __local_buffer_size = __buffer_size;

    // Insert from shared memory buffer into the new container using tiles
    for (auto __tidx = __tile_rank; __tidx < __local_buffer_size; __tidx += __tiles_per_block)
    {
      __container_ref.insert(__tile, __buffer[__tidx]);
    }
    __block.sync();

    __idx += __loop_stride;
  }
}
} // namespace cuda::experimental::cuco::__open_addressing

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_KERNELS_CUH
