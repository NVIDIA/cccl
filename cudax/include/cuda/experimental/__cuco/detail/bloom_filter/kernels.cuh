//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_BLOOM_FILTER_KERNELS_CUH
#define _CUDAX___CUCO_DETAIL_BLOOM_FILTER_KERNELS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>

#include <cuda/experimental/__cuco/detail/utility/cuda.cuh>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::__bloom_filter_ns
{
//! @brief Scalar (cooperative-group size 1) functor adding `first[i]` to the filter.
template <class _InputIt, class _Ref>
struct __add_fn
{
  _InputIt __first;
  _Ref __ref;

  _CCCL_DEVICE_API void operator()(detail::__index_type __idx)
  {
    __ref.__add(*(__first + __idx));
  }
};

template <class _InputIt, class _Ref>
__add_fn(_InputIt, _Ref) -> __add_fn<_InputIt, _Ref>;

//! @brief Scalar (cooperative-group size 1) functor writing `contains(first[i])`.
template <class _InputIt, class _OutputIt, class _Ref>
struct __contains_fn
{
  _InputIt __first;
  _OutputIt __output_begin;
  _Ref __ref;

  _CCCL_DEVICE_API void operator()(detail::__index_type __idx) const
  {
    *(__output_begin + __idx) = __ref.__contains(*(__first + __idx));
  }
};

template <class _InputIt, class _OutputIt, class _Ref>
__contains_fn(_InputIt, _OutputIt, _Ref) -> __contains_fn<_InputIt, _OutputIt, _Ref>;

//! @brief Adds all keys in the range `[first, first + n)` to the filter.
//!
//! @note One thread owns one key; a cooperative group of `_CgSize` threads processes its members'
//! keys one after another so that the words of a single filter block are updated in parallel.
//!
//! @tparam _CgSize Cooperative-group size, i.e. the policy's `add_horizontal_layout`
//! @tparam _BlockSize Number of threads per block
//! @tparam _InputIt Device-accessible random access input key iterator
//! @tparam _Ref Filter implementation type
//!
//! @param __first Beginning of the sequence of keys
//! @param __n Number of keys
//! @param __ref Filter implementation object
template <int _CgSize, int _BlockSize, class _InputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void
__add_n(_InputIt __first, detail::__index_type __n, _Ref __ref)
{
  using __key_type = typename ::cuda::std::iterator_traits<_InputIt>::value_type;

  const auto __idx          = detail::__global_thread_id();
  const auto __group        = ::cooperative_groups::tiled_partition<_CgSize>(::cooperative_groups::this_thread_block());
  const auto __is_full_tile = (static_cast<detail::__index_type>(blockIdx.x) + 1) * _BlockSize <= __n;

  if (__is_full_tile)
  {
    const __key_type& __key = *(__first + __idx);
    __ref.__add_coop(__group, __key);
  }
  else
  {
    __ref.__add_coop(__group, __first, __idx, __idx < __n);
  }
}

//! @brief Tests all keys in the range `[first, first + n)` for presence in the filter.
//!
//! @tparam _CgSize Cooperative-group size, i.e. the policy's `contains_horizontal_layout`
//! @tparam _BlockSize Number of threads per block
//! @tparam _InputIt Device-accessible random access input key iterator
//! @tparam _OutputIt Device-accessible output iterator assignable from `bool`
//! @tparam _Ref Filter implementation type
//!
//! @param __first Beginning of the sequence of keys
//! @param __n Number of keys
//! @param __output_begin Beginning of the sequence of booleans for the presence of each key
//! @param __ref Filter implementation object
template <int _CgSize, int _BlockSize, class _InputIt, class _OutputIt, class _Ref>
_CCCL_KERNEL_ATTRIBUTES _CCCL_LAUNCH_BOUNDS(_BlockSize) void
__contains_n(_InputIt __first, detail::__index_type __n, _OutputIt __output_begin, _Ref __ref)
{
  using __key_type = typename ::cuda::std::iterator_traits<_InputIt>::value_type;

  const auto __idx          = detail::__global_thread_id();
  const auto __group        = ::cooperative_groups::tiled_partition<_CgSize>(::cooperative_groups::this_thread_block());
  const auto __is_full_tile = (static_cast<detail::__index_type>(blockIdx.x) + 1) * _BlockSize <= __n;

  if (__is_full_tile)
  {
    const __key_type& __key   = *(__first + __idx);
    *(__output_begin + __idx) = __ref.__contains_coop(__group, __key);
  }
  else
  {
    const auto __is_valid = __idx < __n;
    const auto __result   = __ref.__contains_coop(__group, __first, __idx, __is_valid);
    if (__is_valid)
    {
      *(__output_begin + __idx) = __result;
    }
  }
}
} // namespace cuda::experimental::cuco::__bloom_filter_ns

_CCCL_DIAG_POP

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_BLOOM_FILTER_KERNELS_CUH
