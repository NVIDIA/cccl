//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DATA_MANIPULATION_COPY
#define __CUDAX_DATA_MANIPULATION_COPY

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__ranges/concepts.h>

#include <cuda/experimental/__data_manipulation/common.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

namespace cuda::experimental
{

template <typename _SrcTy, ::std::size_t _SrcSize, typename _DstTy, ::std::size_t _DstSize>
void __copy_bytes_impl(
  stream_ref __stream, _CUDA_VSTD::span<_SrcTy, _SrcSize> __src, _CUDA_VSTD::span<_DstTy, _DstSize> __dst)
{
  static_assert(!_CUDA_VSTD::is_const_v<_DstTy>, "Copy destination can't be const");
  static_assert(_CUDA_VSTD::is_trivially_copyable_v<_SrcTy> && _CUDA_VSTD::is_trivially_copyable_v<_DstTy>);

  if (__src.size_bytes() > __dst.size_bytes())
  {
    _CUDA_VSTD::__throw_invalid_argument("Copy destination is too small to fit the source data");
  }

  // TODO pass copy direction hint once we have span with properties
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "Failed to perform a copy",
    __dst.data(),
    __src.data(),
    __src.size_bytes(),
    cudaMemcpyDefault,
    __stream.get());
}

_LIBCUDACXX_TEMPLATE(typename _SrcTy, typename _DstTy)
_LIBCUDACXX_REQUIRES(_CUDA_VRANGES::contiguous_range<detail::__as_copy_arg_t<_SrcTy>> _LIBCUDACXX_AND
                       _CUDA_VRANGES::contiguous_range<detail::__as_copy_arg_t<_DstTy>>)
void copy_bytes(stream_ref __stream, _SrcTy&& __src, _DstTy&& __dst)
{
  __copy_bytes_impl(
    __stream,
    _CUDA_VSTD::span(static_cast<detail::__as_copy_arg_t<_SrcTy>>(
      detail::__launch_transform(__stream, _CUDA_VSTD::forward<_SrcTy>(__src)))),
    _CUDA_VSTD::span(static_cast<detail::__as_copy_arg_t<_DstTy>>(
      detail::__launch_transform(__stream, _CUDA_VSTD::forward<_DstTy>(__dst)))));
}

} // namespace cuda::experimental
#endif // __CUDAX_DATA_MANIPULATION_COPY
