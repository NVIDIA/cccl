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
      const typename ::cuda::std::iterator_traits<_InputIt>::value_type __insert_element{*(__first + __idx)};
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
      const typename ::cuda::std::iterator_traits<_InputIt>::value_type __insert_element{*(__first + __idx)};
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
        const typename ::cuda::std::iterator_traits<_InputIt>::value_type __key = *(__first + __idx);
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
        const typename ::cuda::std::iterator_traits<_InputIt>::value_type __key = *(__first + __idx);
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
} // namespace cuda::experimental::cuco::__open_addressing

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_KERNELS_CUH
