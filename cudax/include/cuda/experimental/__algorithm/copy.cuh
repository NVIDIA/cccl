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

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <cuda/experimental/__algorithm/common.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

template <typename _SrcTy, typename _DstTy>
void __copy_bytes_impl(stream_ref __stream, _CUDA_VSTD::span<_SrcTy> __src, _CUDA_VSTD::span<_DstTy> __dst)
{
  static_assert(!_CUDA_VSTD::is_const_v<_DstTy>, "Copy destination can't be const");
  static_assert(_CUDA_VSTD::is_trivially_copyable_v<_SrcTy> && _CUDA_VSTD::is_trivially_copyable_v<_DstTy>);

  if (__src.size_bytes() > __dst.size_bytes())
  {
    _CUDA_VSTD::__throw_invalid_argument("Copy destination is too small to fit the source data");
  }

  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "Failed to perform a copy",
    __dst.data(),
    __src.data(),
    __src.size_bytes(),
    cudaMemcpyDefault,
    __stream.get());
}

//! @brief Launches a bytewise memory copy from source to destination into the provided stream.
//!
//! Both source and destination needs to either be a `contiguous_range` or launch transform to one.
//! They can also implicitly convert to `cuda::std::span`, but the type needs to contain `value_type` member alias.
//! Both source and destination type is required to be trivially copyable.
//!
//! This call might be synchronous if either source or destination is pagable host memory.
//! It will be synchronous if both destination and copy is located in host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __src Source to copy from
//! @param __dst Destination to copy into
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(__valid_1d_copy_fill_argument<_SrcTy> _CCCL_AND __valid_1d_copy_fill_argument<_DstTy>)
void copy_bytes(stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst)
{
  __copy_bytes_impl(
    __stream,
    _CUDA_VSTD::span(__kernel_transform(__launch_transform(__stream, _CUDA_VSTD::forward<_SrcTy>(__src)))),
    _CUDA_VSTD::span(__kernel_transform(__launch_transform(__stream, _CUDA_VSTD::forward<_DstTy>(__dst)))));
}

template <typename _SrcExtents, typename _DstExtents>
[[nodiscard]] bool __copy_bytes_runtime_extents_match(_SrcExtents __src_exts, _DstExtents __dst_exts)
{
  for (typename _SrcExtents::rank_type __i = 0; __i < __src_exts.rank(); __i++)
  {
    if (__src_exts.extent(__i)
        != static_cast<typename _SrcExtents::index_type>(
          __dst_exts.extent((static_cast<typename _DstExtents::rank_type>(__i)))))
    {
      return false;
    }
  }
  return true;
}

template <typename _SrcElem,
          typename _SrcExtents,
          typename _SrcLayout,
          typename _SrcAccessor,
          typename _DstElem,
          typename _DstExtents,
          typename _DstLayout,
          typename _DstAccessor>
void __nd_copy_bytes_impl(stream_ref __stream,
                          _CUDA_VSTD::mdspan<_SrcElem, _SrcExtents, _SrcLayout, _SrcAccessor> __src,
                          _CUDA_VSTD::mdspan<_DstElem, _DstExtents, _DstLayout, _DstAccessor> __dst)
{
  static_assert(_CUDA_VSTD::is_constructible_v<_DstExtents, _SrcExtents>,
                "Multidimensional copy requires both source and destination extents to be compatible");
  static_assert(_CUDA_VSTD::is_same_v<_SrcLayout, _DstLayout>,
                "Multidimensional copy requires both source and destination layouts to match");

  // Check only destination, because the layout of destination is the same as source
  if (!__dst.is_exhaustive())
  {
    _CUDA_VSTD::__throw_invalid_argument("copy_bytes supports only exhaustive mdspans");
  }

  if (!__copy_bytes_runtime_extents_match(__src.extents(), __dst.extents()))
  {
    _CUDA_VSTD::__throw_invalid_argument("Copy destination size differs from the source");
  }

  __copy_bytes_impl(__stream,
                    _CUDA_VSTD::span(__src.data_handle(), __src.mapping().required_span_size()),
                    _CUDA_VSTD::span(__dst.data_handle(), __dst.mapping().required_span_size()));
}

//! @brief Launches a bytewise memory copy from source to destination into the provided stream.
//!
//! Both source and destination needs to either be an instance of `cuda::std::mdspan` or launch transform to
//! one. They can also implicitly convert to `cuda::std::mdspan`, but the type needs to contain `mdspan` template
//! arguments as member aliases named `value_type`, `extents_type`, `layout_type` and `accessor_type`.
//! Resulting mdspan is required to be exhaustive.
//! Both source and destination type is required to be trivially copyable.
//!
//! This call might be synchronous if either source or destination is pagable host memory.
//! It will be synchronous if both destination and copy is located in host memory.
//!
//! @param __stream Stream that the copy should be inserted into
//! @param __src Source to copy from
//! @param __dst Destination to copy into
_CCCL_TEMPLATE(typename _SrcTy, typename _DstTy)
_CCCL_REQUIRES(__valid_nd_copy_fill_argument<_SrcTy> _CCCL_AND __valid_nd_copy_fill_argument<_DstTy>)
void copy_bytes(stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst)
{
  decltype(auto) __src_transformed = __launch_transform(__stream, _CUDA_VSTD::forward<_SrcTy>(__src));
  decltype(auto) __dst_transformed = __launch_transform(__stream, _CUDA_VSTD::forward<_DstTy>(__dst));
  decltype(auto) __src_as_arg      = __kernel_transform(__src_transformed);
  decltype(auto) __dst_as_arg      = __kernel_transform(__dst_transformed);
  __nd_copy_bytes_impl(
    __stream, __as_mdspan_t<decltype(__src_as_arg)>(__src_as_arg), __as_mdspan_t<decltype(__dst_as_arg)>(__dst_as_arg));
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_ALGORITHM_COPY
