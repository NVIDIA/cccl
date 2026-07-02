//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___ALGORITHM_FILL
#define __CUDA___ALGORITHM_FILL

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__algorithm/common.h>
#  include <cuda/__stream/launch_transform.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{
//! @brief Implementation of fill_bytes for span destinations.
//!
//! @tparam _ValueTy The fill value type (1, 2, or 4 bytes).
//! @param __stream Stream to insert the fill into.
//! @param __dst Destination span to fill.
//! @param __value The fill pattern. Each element-sized unit of the destination is set to this value.
//!
//! When `sizeof(_ValueTy) > 1`, the destination size in bytes must be a multiple of `sizeof(_ValueTy)`.
template <typename _ValueTy, typename _DstTy, ::cuda::std::size_t _DstSize>
_CCCL_HOST_API void __fill_bytes_impl(stream_ref __stream, ::cuda::std::span<_DstTy, _DstSize> __dst, _ValueTy __value)
{
  static_assert(!::cuda::std::is_const_v<_DstTy>, "Fill destination can't be const");
  static_assert(::cuda::is_trivially_copyable_v<_DstTy>);

  const auto __size_bytes = __dst.size_bytes();
  if constexpr (sizeof(_ValueTy) > 1)
  {
    if (__size_bytes % sizeof(_ValueTy) != 0)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "fill_bytes destination size in bytes must be a multiple of the fill value size");
    }
  }

  // TODO do a host callback if not device accessible?
  ::cuda::__driver::__memsetAsync(__dst.data(), __value, __size_bytes / sizeof(_ValueTy), __stream.get());
}

//! @brief Implementation of fill_bytes for mdspan destinations.
template <typename _ValueTy, typename _DstElem, typename _DstExtents, typename _DstLayout, typename _DstAccessor>
_CCCL_HOST_API void __fill_bytes_impl(
  stream_ref __stream, ::cuda::std::mdspan<_DstElem, _DstExtents, _DstLayout, _DstAccessor> __dst, _ValueTy __value)
{
  // Check if the mdspan is exhaustive
  if (!__dst.is_exhaustive())
  {
    _CCCL_THROW(::std::invalid_argument, "fill_bytes supports only exhaustive mdspans");
  }

  ::cuda::__detail::__fill_bytes_impl(
    __stream, ::cuda::std::span(__dst.data_handle(), __dst.mapping().required_span_size()), __value);
}
} // namespace __detail

//! @brief Launches an operation to fill the memory with a repeating pattern into the provided stream.
//!
//! The destination needs to be or launch_transform to a `contiguous_range` and convert to `cuda::std::span`.
//! The element type of the destination is required to be trivially copyable.
//!
//! The destination cannot reside in pagable host memory.
//!
//! The fill value can be 1, 2, or 4 bytes wide. The destination size in bytes must be a multiple of
//! the fill value size.
//!
//! @param __stream Stream that the fill should be inserted into
//! @param __dst Destination memory to fill
//! @param __value Value to fill into the destination (1, 2, or 4 bytes)
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__spannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint8_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, ::cuda::std::span(launch_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

//! @overload
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__spannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint16_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, ::cuda::std::span(launch_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

//! @overload
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__spannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint32_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, ::cuda::std::span(launch_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

//! @overload
//! @note This overload accepts mdspan-compatible types.
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__mdspannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint8_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, __as_mdspan(launch_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

//! @overload
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__mdspannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint16_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, __as_mdspan(launch_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

//! @overload
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__mdspannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint32_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, __as_mdspan(launch_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // __CUDA___ALGORITHM_FILL
