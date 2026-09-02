//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_DISPATCH_BY_VECTOR_H
#define _CUDAX__COPY_DISPATCH_BY_VECTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cuda/experimental/__copy/tensor_copy_utils.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Compute the maximum vector access width in bytes for a pair of raw tensors.
//!
//! Takes the minimum of the source alignment, destination alignment, and the GPU architecture's
//! maximum vector width.
//!
//! @param[in] __src Source raw tensor
//! @param[in] __dst Destination raw tensor
//! @return Maximum safe vector access width in bytes
template <typename _SrcExtentT,
          typename _SrcStrideT,
          typename _TpSrc,
          typename _DstExtentT,
          typename _DstStrideT,
          typename _TpDst,
          ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__vector_size_bytes(const __raw_tensor<_SrcExtentT, _SrcStrideT, _TpSrc, _MaxRank>& __src,
                    const __raw_tensor<_DstExtentT, _DstStrideT, _TpDst, _MaxRank>& __dst) noexcept
{
  return ::cuda::std::min(
    {::cuda::experimental::__max_alignment(__src),
     ::cuda::experimental::__max_alignment(__dst),
     ::cuda::experimental::__max_gpu_arch_vector_size()});
}

template <int _VectorSize>
constexpr auto __const_vector_size = ::cuda::std::integral_constant<int, _VectorSize>{};

//! @brief Dispatch a copy operation with the optimal vectorized element type.
//!
//! Computes the maximum safe vector width from the source and destination tensors, reshapes both
//! tensors to that vector type via @ref __reshape_vectorized, and invokes @p __op with the reshaped tensors.
//!
//! @param[in] __src Source raw tensor descriptor
//! @param[in] __dst Destination raw tensor descriptor
//! @param[in] __op  Callable invoked with the reshaped source and destination tensors
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
  _Op __op) noexcept
{
  namespace cudax              = ::cuda::experimental;
  const auto __call_vectorized = [&](auto __const_vector_size) {
    const auto __src_recast = cudax::__reshape_vectorized<__const_vector_size>(__src);
    const auto __dst_recast = cudax::__reshape_vectorized<__const_vector_size>(__dst);
    __op(__src_recast, __dst_recast);
  };
  const auto __vector_size_bytes = cudax::__vector_size_bytes(__src, __dst);
// 32-bytes aligned vector types have been introduced in CTK 13.0
#  if _CCCL_CTK_AT_LEAST(13, 0)
  static_assert(sizeof(_TpIn) <= 32);
  if constexpr (sizeof(_TpIn) <= 32)
  {
    if (__vector_size_bytes == 32)
    {
      __call_vectorized(__const_vector_size<32>);
      return;
    }
  }
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  static_assert(sizeof(_TpIn) <= 16);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  if constexpr (sizeof(_TpIn) <= 16)
  {
    if (__vector_size_bytes == 16)
    {
      __call_vectorized(__const_vector_size<16>);
      return;
    }
  }
  if constexpr (sizeof(_TpIn) <= 8)
  {
    if (__vector_size_bytes == 8)
    {
      __call_vectorized(__const_vector_size<8>);
      return;
    }
  }
  if constexpr (sizeof(_TpIn) <= 4)
  {
    if (__vector_size_bytes == 4)
    {
      __call_vectorized(__const_vector_size<4>);
      return;
    }
  }
  if constexpr (sizeof(_TpIn) <= 2)
  {
    if (__vector_size_bytes == 2)
    {
      __call_vectorized(__const_vector_size<2>);
      return;
    }
  }
  if constexpr (sizeof(_TpIn) <= 1)
  {
    __call_vectorized(__const_vector_size<1>);
  }
  // no fallthrough (sizeof(T) is never 0)
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_DISPATCH_BY_VECTOR_H
