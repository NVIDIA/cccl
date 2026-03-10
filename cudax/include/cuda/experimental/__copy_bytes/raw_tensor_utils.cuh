//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_RAW_TENSOR_UTILS_H
#define __CUDAX_COPY_RAW_TENSOR_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__utility/in_range.h>
#  include <cuda/std/__algorithm/all_of.h>
#  include <cuda/std/__algorithm/is_sorted.h>
#  include <cuda/std/__cstddef/types.h>

#  include <cuda/experimental/__copy_bytes/abs_integer.cuh>
#  include <cuda/experimental/__copy_bytes/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Check whether the first `__rank` strides are in non-descending order.
//!
//! @pre `__tensor.__rank` is in [0, _MaxRank].
//!
//! @param[in] __tensor Raw tensor to inspect
//! @return true if strides[0..rank) are non-descending
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool __has_sorted_strides(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  namespace cudax = ::cuda::experimental;
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, ::cuda::std::size_t{0}, _MaxRank), "Invalid tensor rank");
  return ::cuda::std::is_sorted(
    __tensor.__strides.cbegin(), __tensor.__strides.cbegin() + __tensor.__rank, [](auto __a, auto __b) {
      return cudax::__abs_integer(__a) < cudax::__abs_integer(__b);
    });
}

//! @brief Check whether every active mode has shape strictly greater than 1.
//!
//! @pre `__tensor.__rank` is in [0, _MaxRank].
//!
//! @param[in] __tensor Raw tensor to inspect
//! @return true if all shapes in [0, rank) are > 1; true for rank 0
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool __has_no_extent1_modes(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, ::cuda::std::size_t{0}, _MaxRank), "Invalid tensor rank");
  // clang-format off
  return ::cuda::std::all_of(__tensor.__extents.cbegin(), __tensor.__extents.cbegin() + __tensor.__rank,
    [](auto __extent) {
      return __extent > 1;
    }
  ); // clang-format on
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_RAW_TENSOR_UTILS_H
