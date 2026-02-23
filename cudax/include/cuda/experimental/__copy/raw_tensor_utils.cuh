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

#  include <cuda/experimental/__copy/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool __has_sorted_strides(const __raw_tensor<_Tp, _MaxRank>& __tensor) noexcept
{
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, 0, _MaxRank - 1), "Invalid tensor rank");
  return ::cuda::std::is_sorted(__tensor.__strides.cbegin(), __tensor.__strides.cbegin() + __tensor.__rank);
}

template <typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API bool __has_no_extent1_modes(const __raw_tensor<_Tp, _MaxRank>& __tensor) noexcept
{
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, 0, _MaxRank - 1), "Invalid tensor rank");
  // clang-format off
  return ::cuda::std::all_of(__tensor.__shapes.cbegin(), __tensor.__shapes.cbegin() + __tensor.__rank,
    [](auto __shape) {
      return __shape > 1;
    }
  ); // clang-format on
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_RAW_TENSOR_UTILS_H
