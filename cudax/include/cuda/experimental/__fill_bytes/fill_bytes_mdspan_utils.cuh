//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX___FILL_BYTES_FILL_BYTES_MDSPAN_UTILS_H
#define __CUDAX___FILL_BYTES_FILL_BYTES_MDSPAN_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/is_signed.h>

#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Flips negative-stride modes to equivalent positive-stride modes.
//!
//! @param[in,out] __tensor Raw tensor to flip negative strides for
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void
__flip_negative_strides_single([[maybe_unused]] __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  if constexpr (::cuda::std::is_signed_v<_StrideT>)
  {
    using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
    using __rank_t       = typename __raw_tensor_t::__rank_t;
    for (__rank_t __i = 0; __i < __tensor.__rank; ++__i)
    {
      if (__tensor.__strides[__i] < 0)
      {
        const auto __extent = static_cast<_StrideT>(__tensor.__extents[__i] - 1);
        __tensor.__data += __extent * __tensor.__strides[__i];
        _CCCL_ASSERT(__tensor.__strides[__i] != ::cuda::std::numeric_limits<_StrideT>::min(),
                     "cudax::flip_negative_strides_single: stride is at min value");
        __tensor.__strides[__i] = -__tensor.__strides[__i];
      }
    }
  }
}

//! @brief Merges adjacent modes that are contiguous in the destination tensor.
//!
//! @param[in,out] __tensor Raw tensor to coalesce modes for
template <typename _ExtentT, typename _StrideT, typename _Tp, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __coalesce_single(__raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>& __tensor) noexcept
{
  if (__tensor.__rank <= 1)
  {
    return;
  }
  using __raw_tensor_t = __raw_tensor<_ExtentT, _StrideT, _Tp, _MaxRank>;
  using __rank_t       = typename __raw_tensor_t::__rank_t;
  __rank_t __out_rank  = 1;
  for (__rank_t __i = 1; __i < __tensor.__rank; ++__i)
  {
    const auto __prev_extent = static_cast<_StrideT>(__tensor.__extents[__out_rank - 1]);
    if (__prev_extent * __tensor.__strides[__out_rank - 1] == __tensor.__strides[__i])
    {
      __tensor.__extents[__out_rank - 1] *= __tensor.__extents[__i];
      continue;
    }
    __tensor.__extents[__out_rank] = __tensor.__extents[__i];
    __tensor.__strides[__out_rank] = __tensor.__strides[__i];
    ++__out_rank;
  }
  for (__rank_t __i = __out_rank; __i < _MaxRank; ++__i)
  {
    __tensor.__extents[__i] = _ExtentT{1};
  }
  __tensor.__rank = __out_rank;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX___FILL_BYTES_FILL_BYTES_MDSPAN_UTILS_H
