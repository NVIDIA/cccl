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

#include <cuda/std/__concepts/concept_macros.h>

#include <cuda/experimental/__algorithm/common.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

template <typename _DstTy, ::std::size_t _DstSize>
void __fill_bytes_impl(stream_ref __stream, _CUDA_VSTD::span<_DstTy, _DstSize> __dst, uint8_t __value)
{
  static_assert(!_CUDA_VSTD::is_const_v<_DstTy>, "Fill destination can't be const");
  static_assert(_CUDA_VSTD::is_trivially_copyable_v<_DstTy>);

  // TODO do a host callback if not device accessible?
  _CCCL_TRY_CUDA_API(
    ::cudaMemsetAsync, "Failed to perform a fill", __dst.data(), __value, __dst.size_bytes(), __stream.get());
}

//! @brief Launches an operation to bytewise fill the memory into the provided stream.
//!
//! Destination needs to either be a `contiguous_range` or launch transform
//! into one. It can also implicitly convert to `cuda::std::span`, but it needs to contain `value_type` member alias.
//! Destination type is required to be trivially copyable.
//!
//! Destination can't reside in pagable host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __dst Destination memory to fill
//! @param __value Value to fill into every byte in the destination
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__valid_1d_copy_fill_argument<_DstTy>)
void fill_bytes(stream_ref __stream, _DstTy&& __dst, uint8_t __value)
{
  __fill_bytes_impl(
    __stream,
    _CUDA_VSTD::span(__kernel_transform(__launch_transform(__stream, _CUDA_VSTD::forward<_DstTy>(__dst)))),
    __value);
}

//! @brief Launches an operation to bytewise fill the memory into the provided stream.
//!
//! Destination needs to either be an instance of `cuda::std::mdspan` or launch transform
//! into one. It can also implicitly convert to `cuda::std::mdspan`, but the type needs to contain `mdspan` template
//! arguments as member aliases named `value_type`, `extents_type`, `layout_type` and `accessor_type`.
//! Resulting mdspan is required to be exhaustive.
//! Destination type is required to be trivially copyable.
//!
//! Destination can't reside in pagable host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __dst Destination memory to fill
//! @param __value Value to fill into every byte in the destination
_CCCL_TEMPLATE(typename _DstTy)
_CCCL_REQUIRES(__valid_nd_copy_fill_argument<_DstTy>)
void fill_bytes(stream_ref __stream, _DstTy&& __dst, uint8_t __value)
{
  decltype(auto) __dst_transformed = __launch_transform(__stream, _CUDA_VSTD::forward<_DstTy>(__dst));
  decltype(auto) __dst_as_arg      = __kernel_transform(__dst_transformed);
  auto __dst_mdspan                = __as_mdspan_t<decltype(__dst_as_arg)>(__dst_as_arg);

  if (!__dst_mdspan.is_exhaustive())
  {
    _CUDA_VSTD::__throw_invalid_argument("fill_bytes supports only exhaustive mdspans");
  }

  __fill_bytes_impl(
    __stream, _CUDA_VSTD::span(__dst_mdspan.data_handle(), __dst_mdspan.mapping().required_span_size()), __value);
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_ALGORITHM_FILL
