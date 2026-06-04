//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_UTILS_CUH
#define _CUDAX___CUCO___DETAIL_UTILS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Counts least significant bits in a 32-bit value.
_CCCL_DEVICE inline ::cuda::std::int32_t
__count_least_significant_bits(::cuda::std::uint32_t __x, ::cuda::std::int32_t __n)
{
  if (__n <= 0)
  {
    return 0;
  }
  if (__n >= 32)
  {
    return __popc(__x);
  }
  return __popc(__x & ((1u << __n) - 1u));
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_UTILS_CUH
