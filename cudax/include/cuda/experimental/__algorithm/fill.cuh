//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ALGORITHM_FILL
#define __CUDAX_ALGORITHM_FILL

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__algorithm/fill.h>
#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__stream/device_transform.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Launches an operation to bytewise fill the memory into the provided stream.
//!
//! The destination needs to either be a `contiguous_range` or transform into one. It can
//! also implicitly convert to `cuda::std::span`, but it needs to contain a `value_type`
//! member alias. The element type of the destination is required to be trivially
//! copyable.
//!
//! The destination cannot reside in pagable host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __dst Destination memory to fill
//! @param __value Value to fill into every byte in the destination
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(::cuda::__spannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(::cuda::stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint8_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, ::cuda::std::span(device_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}

//! @brief Launches an operation to bytewise fill the memory into the provided stream.
//!
//! Destination needs to either be an instance of `cuda::std::mdspan` or transform into
//! one. It can also implicitly convert to `cuda::std::mdspan`, but the type needs to
//! contain `mdspan` template arguments as member aliases named `value_type`,
//! `extents_type`, `layout_type` and `accessor_type`. The resulting mdspan is required to
//! be exhaustive. The element type of the destination is required to be trivially
//! copyable.
//!
//! The destination cannot reside in pagable host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __dst Destination memory to fill
//! @param __value Value to fill into every byte in the destination
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(::cuda::__mdspannable<transformed_device_argument_t<_DstTy>>)
_CCCL_HOST_API void fill_bytes(::cuda::stream_ref __stream, _DstTy&& __dst, ::cuda::std::uint8_t __value)
{
  ::cuda::__detail::__fill_bytes_impl(
    __stream, ::cuda::__as_mdspan(device_transform(__stream, ::cuda::std::forward<_DstTy>(__dst))), __value);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_ALGORITHM_FILL
