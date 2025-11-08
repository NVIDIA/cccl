//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ALGORITHM_COPY
#define __CUDAX_ALGORITHM_COPY

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__algorithm/copy.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <cuda/experimental/__stream/device_transform.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Launches a bytewise memory copy from source to destination into the provided
//! stream.
//!
//! Both source and destination needs to be or device_transform to a type that is a `contiguous_range` and converts to
//! `cuda::std::span`. The element types of both the source and destination range is required to be trivially copyable.
//!
//! This call might be synchronous if either source or destination is pagable host memory.
//! It will be synchronous if both destination and copy is located in host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __src Source to copy from
//! @param __dst Destination to copy into
//! @param __config Configuration for the copy
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(::cuda::__spannable<transformed_device_argument_t<_SrcTy>>
                 _CCCL_AND ::cuda::__spannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void
copy_bytes(::cuda::stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst, copy_configuration __config = {})
{
  ::cuda::__detail::__copy_bytes_impl(
    __stream,
    ::cuda::std::span(device_transform(__stream, ::cuda::std::forward<_SrcTy>(__src))),
    ::cuda::std::span(device_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))),
    __config);
}

//! @brief Launches a bytewise memory copy from source to destination into the provided
//! stream.
//!
//! Both source and destination needs to be or device_transform to an instance of `cuda::std::mdspan`.
//! They can also implicitly convert to `cuda::std::mdspan`, but the
//! type needs to contain `mdspan` template arguments as member aliases named
//! `value_type`, `extents_type`, `layout_type` and `accessor_type`. The resulting mdspan
//! is required to be exhaustive. The element types of both the source and destination
//! type are required to be trivially copyable.
//!
//! This call might be synchronous if either source or destination is pagable host memory.
//! It will be synchronous if both destination and copy is located in host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __src Source to copy from
//! @param __dst Destination to copy into
//! @param __config Configuration for the copy
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(::cuda::__mdspannable<transformed_device_argument_t<_SrcTy>>
                 _CCCL_AND ::cuda::__mdspannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void
copy_bytes(::cuda::stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst, copy_configuration __config = {})
{
  ::cuda::__detail::__copy_bytes_impl(
    __stream,
    ::cuda::__as_mdspan(device_transform(__stream, ::cuda::std::forward<_SrcTy>(__src))),
    ::cuda::__as_mdspan(device_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))),
    __config);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_ALGORITHM_COPY
