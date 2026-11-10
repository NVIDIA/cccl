//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_CLAMP_H
#define _CUDA___CMATH_CLAMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/min_max.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/cast.h>
#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Clamps a value to the range `[__lo, __hi]`, computed as `cuda::max(cuda::min(__v, __hi), __lo)`. For
//! floating-point types, a NaN boundary is ignored and a NaN `__v` results in `__hi`, or in `__lo` if `__hi` is NaN.
//! @param[in] __v The value to clamp.
//! @param[in] __lo The lower boundary.
//! @param[in] __hi The upper boundary.
//! @pre `__hi` is not less than `__lo`.
//! @return `__lo` if `__v` is less than `__lo`, `__hi` if `__hi` is less than `__v`, otherwise `__v`.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_TRIVIAL_API constexpr _Tp clamp(_Tp __v, _Tp __lo, _Tp __hi) noexcept
{
  // The branches are identical when assertions are disabled.
  // NOLINTBEGIN(bugprone-branch-clone)
  if constexpr (::cuda::std::__is_ext_nv_fp_v<_Tp>)
  {
    _CCCL_ASSERT(!(::cuda::std::__fp_cast<float>(__hi) < ::cuda::std::__fp_cast<float>(__lo)),
                 "Bad bounds passed to cuda::clamp");
  }
  else
  {
    // Ordered comparisons with NaN are not constant expressions
    _CCCL_ASSERT(::cuda::std::isnan(__lo) || ::cuda::std::isnan(__hi) || !(__hi < __lo),
                 "Bad bounds passed to cuda::clamp");
  }
  // NOLINTEND(bugprone-branch-clone)
  return ::cuda::max(::cuda::min(__v, __hi), __lo);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_CLAMP_H
