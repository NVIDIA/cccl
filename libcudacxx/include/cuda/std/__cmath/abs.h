//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_ABS_H
#define _LIBCUDACXX___CMATH_ABS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_fabs) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_FABSF(...) __builtin_fabsf(__VA_ARGS__)
#  define _CCCL_BUILTIN_FABS(...)  __builtin_fabs(__VA_ARGS__)
#  define _CCCL_BUILTIN_FABSL(...) __builtin_fabsl(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_fabs)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// fabs

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto fabs(_Tp __x) noexcept
{
  if constexpr (!numeric_limits<_Tp>::is_signed)
  {
    if constexpr (is_integral_v<_Tp>)
    {
      return static_cast<double>(__x);
    }
    else
    {
      return __x;
    }
  }
  else if constexpr (is_integral_v<_Tp>)
  {
    return __x < 0 ? -static_cast<double>(__x) : static_cast<double>(__x);
  }
  else
  {
#if !_CCCL_HAS_CONSTEXPR_BIT_CAST() && _CCCL_COMPILER(GCC)
    if constexpr (is_same_v<_Tp, float>)
    {
      return _CCCL_BUILTIN_FABSF(__x);
    }
    else if constexpr (is_same_v<_Tp, double>)
    {
      return _CCCL_BUILTIN_FABS(__x);
    }
#  if _CCCL_HAS_LONG_DOUBLE()
    else if constexpr (is_same_v<_Tp, long double>)
    {
      return _CCCL_BUILTIN_FABSL(__x);
    }
#  endif // _CCCL_HAS_LONG_DOUBLE()
#endif // !_CCCL_HAS_CONSTEXPR_BIT_CAST() && _CCCL_COMPILER(GCC)
    // We cannot use `abs.f16` or `abs.bf16` because it is not IEEE 754 compliant, see docs
    const auto __val = _CUDA_VSTD::__fp_get_storage(__x) & __fp_exp_mant_mask_of_v<_Tp>;
    return _CUDA_VSTD::__fp_from_storage<_Tp>(static_cast<__fp_storage_of_t<_Tp>>(__val));
  }
}

[[nodiscard]] _CCCL_API inline constexpr float fabsf(float __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API inline constexpr long double fabsl(long double __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

// abs

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr auto abs(_Tp __x) noexcept
{
  return _CUDA_VSTD::fabs(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CMATH_ABS_H
