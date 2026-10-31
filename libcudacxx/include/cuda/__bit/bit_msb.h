//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BIT_MSB_H
#define _CUDA___BIT_BIT_MSB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/integral.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Finds the position of the most significant set bit of \p __value
//! @param __value The unsigned integer value to scan
//! @return The zero-based index of the most significant set bit of \p __value, i.e. floor(log2(value)),
//!         or -1 if \p __value is zero
//! @note bit_msb is the most-significant counterpart to bit_ffs (find first set). It forwards to
//!       cuda::std::__bit_log2, which lowers to the optimal find-leading-bit code (ptx::bfind on device,
//!       a countl_zero based path on host)
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int bit_msb(const _Tp __value) noexcept
{
  const auto __ret = static_cast<int>(::cuda::std::__bit_log2(__value));
  _CCCL_ASSUME(__ret >= -1 && __ret < ::cuda::std::numeric_limits<_Tp>::digits);
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BIT_MSB_H
