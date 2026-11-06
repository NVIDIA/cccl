//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_MIN_MAX_H
#define _CUDA___CMATH_MIN_MAX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Returns the smaller of two values. Floating-point values are compared like with `cuda::std::fmin`, so NaN is
//! only returned if both values are NaN.
//! @param[in] __a The first value.
//! @param[in] __b The second value.
//! @return The smaller of `__a` and `__b`.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_TRIVIAL_API constexpr _Tp min(_Tp __a, _Tp __b) noexcept
{
  if constexpr (::cuda::is_floating_point_v<_Tp>)
  {
    return ::cuda::std::fmin(__a, __b);
  }
  else
  {
    return (__b < __a) ? __b : __a;
  }
}

//! @brief Returns the larger of two values. Floating-point values are compared like with `cuda::std::fmax`, so NaN is
//! only returned if both values are NaN.
//! @param[in] __a The first value.
//! @param[in] __b The second value.
//! @return The larger of `__a` and `__b`.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_TRIVIAL_API constexpr _Tp max(_Tp __a, _Tp __b) noexcept
{
  if constexpr (::cuda::is_floating_point_v<_Tp>)
  {
    return ::cuda::std::fmax(__a, __b);
  }
  else
  {
    return (__a < __b) ? __b : __a;
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_MIN_MAX_H
