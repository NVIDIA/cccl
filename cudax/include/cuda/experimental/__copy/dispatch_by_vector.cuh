//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_VECTORIZED_H
#define _CUDAX__COPY_VECTORIZED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__cstddef/types.h>

#  include <cuda/experimental/__copy/copy_optimized.cuh>
#  include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <int _VectorSize>
constexpr auto __const_vector_size = ::cuda::std::integral_constant<int, _VectorSize>{};

template <typename _ExtentT,
          typename _StrideTIn,
          typename _StrideTOut,
          typename _TpIn,
          typename _TpOut,
          ::cuda::std::size_t _Rank,
          typename _Op>
_CCCL_HOST_API void __dispatch_by_vector_size(
  const __raw_tensor<_ExtentT, _StrideTIn, _TpIn, _Rank>& __src,
  const __raw_tensor<_ExtentT, _StrideTOut, _TpOut, _Rank>& __dst,
  ::cuda::std::size_t __vector_size_bytes,
  _Op __op) noexcept
{
  const auto __call_vectorized = [&](auto __const_vector_size) {
    const auto __src_recast = ::cuda::experimental::__reshape_vectorized<__const_vector_size>(__src);
    const auto __dst_recast = ::cuda::experimental::__reshape_vectorized<__const_vector_size>(__dst);
    __op(__src_recast, __dst_recast);
  };
#  if _CCCL_CTK_AT_LEAST(13, 0)
  if constexpr (sizeof(_TpIn) <= 32)
  {
    if (__vector_size_bytes == 32)
    {
      __call_vectorized(__const_vector_size<32>);
    }
  }
  else
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
    if constexpr (sizeof(_TpIn) <= 16)
    {
      if (__vector_size_bytes == 16)
      {
        __call_vectorized(__const_vector_size<16>);
      }
    }
    else if constexpr (sizeof(_TpIn) <= 8)
    {
      if (__vector_size_bytes == 8)
      {
        __call_vectorized(__const_vector_size<8>);
      }
    }
    else if constexpr (sizeof(_TpIn) <= 4)
    {
      if (__vector_size_bytes == 4)
      {
        __call_vectorized(__const_vector_size<4>);
      }
    }
    else if constexpr (sizeof(_TpIn) <= 2)
    {
      if (__vector_size_bytes == 2)
      {
        __call_vectorized(__const_vector_size<2>);
      }
    }
    else
    {
      __call_vectorized(__const_vector_size<1>);
    }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_VECTORIZED_H
