// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_MIN_MAX_H
#define _LIBCUDACXX___CMATH_MIN_MAX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/limits>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// fmax
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API _CCCL_CONSTEXPR_BIT_CAST auto fmax(_Tp __x, _Tp __y) noexcept
{
  using __ret = conditional_t<is_integral_v<_Tp>, double, _Tp>;
  if constexpr (!is_integral_v<_Tp>)
  {
    if (_CUDA_VSTD::isnan(__x))
    {
      return __y;
    }
    else if (_CUDA_VSTD::isnan(__y))
    {
      return __x;
    }
  }
  return __x < __y ? static_cast<__ret>(__y) : static_cast<__ret>(__x);
}

[[nodiscard]] _CCCL_API inline float fmaxf(float __x, float __y) noexcept
{
  return _CUDA_VSTD::fmax(__x, __y);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fmaxl(long double __x, long double __y) noexcept
{
  return _CUDA_VSTD::fmax(__x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp> _CCCL_AND __is_extended_arithmetic_v<_Up>)
[[nodiscard]] _CCCL_API _CCCL_CONSTEXPR_BIT_CAST __promote_t<_Tp, _Up> fmax(_Tp __x, _Up __y) noexcept
{
  using __result_type = __promote_t<_Tp, _Up>;
  static_assert(!(is_same_v<_Tp, __result_type> && is_same_v<_Up, __result_type>) );
  return _CUDA_VSTD::fmax(static_cast<__result_type>(__x), static_cast<__result_type>(__y));
}

// fmin
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API _CCCL_CONSTEXPR_BIT_CAST auto fmin(_Tp __x, _Tp __y) noexcept
{
  using __ret = conditional_t<is_integral_v<_Tp>, double, _Tp>;
  if constexpr (!is_integral_v<_Tp>)
  {
    if (_CUDA_VSTD::isnan(__x))
    {
      return __y;
    }
    else if (_CUDA_VSTD::isnan(__y))
    {
      return __x;
    }
  }
  return __y < __x ? static_cast<__ret>(__y) : static_cast<__ret>(__x);
}

[[nodiscard]] _CCCL_API inline float fminf(float __x, float __y) noexcept
{
  return _CUDA_VSTD::fmin(__x, __y);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline long double fminl(long double __x, long double __y) noexcept
{
  return _CUDA_VSTD::fmin(__x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp> _CCCL_AND __is_extended_arithmetic_v<_Up>)
[[nodiscard]] _CCCL_API _CCCL_CONSTEXPR_BIT_CAST __promote_t<_Tp, _Up> fmin(_Tp __x, _Up __y) noexcept
{
  using __result_type = __promote_t<_Tp, _Up>;
  static_assert(!(is_same_v<_Tp, __result_type> && is_same_v<_Up, __result_type>) );
  return _CUDA_VSTD::fmin(static_cast<__result_type>(__x), static_cast<__result_type>(__y));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_MIN_MAX_H
