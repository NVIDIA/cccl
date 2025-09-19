//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___ALGORITHM_COPY_H
#define __CUDA___ALGORITHM_COPY_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__algorithm/common.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/mdspan>
#  include <cuda/std/span>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
template <typename _SrcTy, typename _DstTy>
_CCCL_HOST_API void
__copy_bytes_impl(stream_ref __stream, ::cuda::std::span<_SrcTy> __src, ::cuda::std::span<_DstTy> __dst)
{
  static_assert(!::cuda::std::is_const_v<_DstTy>, "Copy destination can't be const");
  static_assert(::cuda::std::is_trivially_copyable_v<_SrcTy> && ::cuda::std::is_trivially_copyable_v<_DstTy>);

  if (__src.size_bytes() > __dst.size_bytes())
  {
    ::cuda::std::__throw_invalid_argument("Copy destination is too small to fit the source data");
  }

  ::cuda::__driver::__memcpyAsync(__dst.data(), __src.data(), __src.size_bytes(), __stream.get());
}

template <typename _SrcElem,
          typename _SrcExtents,
          typename _SrcLayout,
          typename _SrcAccessor,
          typename _DstElem,
          typename _DstExtents,
          typename _DstLayout,
          typename _DstAccessor>
_CCCL_HOST_API void __copy_bytes_impl(stream_ref __stream,
                                      ::cuda::std::mdspan<_SrcElem, _SrcExtents, _SrcLayout, _SrcAccessor> __src,
                                      ::cuda::std::mdspan<_DstElem, _DstExtents, _DstLayout, _DstAccessor> __dst)
{
  static_assert(::cuda::std::is_constructible_v<_DstExtents, _SrcExtents>,
                "Multidimensional copy requires both source and destination extents to be compatible");
  static_assert(::cuda::std::is_same_v<_SrcLayout, _DstLayout>,
                "Multidimensional copy requires both source and destination layouts to match");

  // Check only destination, because the layout of destination is the same as source
  if (!__dst.is_exhaustive())
  {
    ::cuda::std::__throw_invalid_argument("copy_bytes supports only exhaustive mdspans");
  }

  if (__src.extents() != __dst.extents())
  {
    ::cuda::std::__throw_invalid_argument("Copy destination size differs from the source");
  }

  ::cuda::__detail::__copy_bytes_impl(
    __stream,
    ::cuda::std::span(__src.data_handle(), __src.mapping().required_span_size()),
    ::cuda::std::span(__dst.data_handle(), __dst.mapping().required_span_size()));
}
} // namespace __detail

//! @brief Launches a bytewise memory copy from source to destination into the provided
//! stream.
//!
//! Both source and destination needs to be a `contiguous_range` and convert to
//! `cuda::std::span`. The element types of both the source and destination range is
//! required to be trivially copyable.
//!
//! This call might be synchronous if either source or destination is pagable host memory.
//! It will be synchronous if both destination and copy is located in host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __src Source to copy from
//! @param __dst Destination to copy into
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(__spannable<_SrcTy> _CCCL_AND __spannable<_DstTy>)
_CCCL_HOST_API void copy_bytes(stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst)
{
  ::cuda::__detail::__copy_bytes_impl(
    __stream,
    ::cuda::std::span(::cuda::std::forward<_SrcTy>(__src)),
    ::cuda::std::span(::cuda::std::forward<_DstTy>(__dst)));
}

//! @brief Launches a bytewise memory copy from source to destination into the provided
//! stream.
//!
//! Both source and destination needs to be an instance of `cuda::std::mdspan`.
//! They can also convert to `cuda::std::mdspan`, but the type needs to contain
//! `mdspan` template arguments as member aliases named `value_type`, `extents_type`,
//! `layout_type` and `accessor_type`. The resulting mdspan is required to be
//! exhaustive. The element types of both the source and destination type are
//! required to be trivially copyable.
//!
//! This call might be synchronous if either source or destination is pagable host memory.
//! It will be synchronous if both destination and copy is located in host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __src Source to copy from
//! @param __dst Destination to copy into
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(__mdspannable<_SrcTy> _CCCL_AND __mdspannable<_DstTy>)
_CCCL_HOST_API void copy_bytes(stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst)
{
  ::cuda::__detail::__copy_bytes_impl(
    __stream,
    ::cuda::__as_mdspan(::cuda::std::forward<_SrcTy>(__src)),
    ::cuda::__as_mdspan(::cuda::std::forward<_DstTy>(__dst)));
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // __CUDA___ALGORITHM_COPY_H
