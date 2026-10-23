// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_MIN_MAX_H
#define _CUDA_STD___CMATH_MIN_MAX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// maps the bits of a double to an int64_t in IEEE totalOrder: -0 < +0, NaNs are placed beyond +-inf
[[nodiscard]] _CCCL_API constexpr int64_t __fp64_to_total_order_int(double __x) noexcept
{
  const auto __x_as_int = static_cast<int64_t>(::cuda::std::__fp_get_storage(__x));
  return __x_as_int < 0 ? __x_as_int ^ ::cuda::std::numeric_limits<int64_t>::max() : __x_as_int;
}

// fmax (_IsMax) / fmin for fp64 with integer instructions only
// -0 < +0 is preserved
// single NaN is ignored; two NaNs return __x quieted
template <bool _IsMax>
[[nodiscard]] _CCCL_API constexpr double __fp64_fmax_fmin_int(double __x, double __y) noexcept
{
  constexpr auto __quiet_bit = uint64_t{0x0008'0000'0000'0000}; // top mantissa bit (for quiet NaNs)
  const auto __x_bits        = ::cuda::std::__fp_get_storage(__x);
  const auto __y_bits        = ::cuda::std::__fp_get_storage(__y);
  const bool __x_nan         = ::cuda::std::isnan(__x);
  const bool __y_nan         = ::cuda::std::isnan(__y);
  const bool __x_lt_y = ::cuda::std::__fp64_to_total_order_int(__x) < ::cuda::std::__fp64_to_total_order_int(__y);
  const bool __pick_y = __x_nan ? !__y_nan : (!__y_nan && __x_lt_y == _IsMax);
  const auto __result = __pick_y ? __y_bits : __x_bits;
  return ::cuda::std::__fp_from_storage<double>((__x_nan && __y_nan) ? (__result | __quiet_bit) : __result);
}

/***********************************************************************************************************************
 * fmax
 **********************************************************************************************************************/

// We do explicitly also enable GCC here, because that makes the condition below simpler
#if _CCCL_CHECK_BUILTIN(builtin_fmax) || _CCCL_COMPILER(GCC)
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_API _Tp __with_builtin_fmax(_Tp __x, _Tp __y) noexcept
{
#  if _CCCL_COMPILER(CLANG) && _CCCL_HOST_ARCH(ARM64) && _CCCL_HOST_COMPILATION()
  // Clang lowers fmax to AArch64 fmaxnm, which returns quiet NaN for signaling NaNs,
  // See https://github.com/llvm/llvm-project/issues/176624
  if (::cuda::std::isnan(__x))
  {
    return __y;
  }
  if (::cuda::std::isnan(__y))
  {
    return __x;
  }
#  endif // _CCCL_COMPILER(CLANG) && _CCCL_HOST_ARCH(ARM64) && _CCCL_HOST_COMPILATION()
  if constexpr (is_same_v<_Tp, float>)
  {
    return __builtin_fmaxf(__x, __y);
  }
  else if constexpr (is_same_v<_Tp, double>)
  {
    return __builtin_fmax(__x, __y);
  }
#  if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (is_same_v<_Tp, long double>)
  {
    return __builtin_fmaxl(__x, __y);
  }
#  endif // _CCCL_HAS_LONG_DOUBLE()
}

#  define _CCCL_USE_BUILTIN_FMAX() 1
#else //
#  define _CCCL_USE_BUILTIN_FMAX() 0
#endif // _CCCL_BUILTIN_FABSF

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr conditional_t<is_integral_v<_Tp>, double, _Tp> fmax(_Tp __x, _Tp __y) noexcept
{
#if _CCCL_HAS_NVFP16()
  // The half and bfloat16 branches can become identical under some CUDA Toolkit versions.
  // NOLINTBEGIN(bugprone-branch-clone)
  if constexpr (is_same_v<_Tp, ::__half>)
  {
#  if _CCCL_CTK_AT_LEAST(12, 2)
    return ::__hmax(__x, __y);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return ::__hmax(__x, __y);),
                      (return ::__float2half(::cuda::std::fmax(::__half2float(__x), ::__half2float(__y)));))
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
  }
  // NOLINTEND(bugprone-branch-clone)
  else
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
    if constexpr (is_same_v<_Tp, ::__nv_bfloat16>)
  {
#  if _CCCL_CTK_AT_LEAST(12, 2)
    return ::__hmax(__x, __y);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return ::__hmax(__x, __y);),
                      (return ::__float2bfloat16(::cuda::std::fmax(::__bfloat162float(__x), ::__bfloat162float(__y)));))
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
  }
  else
#endif // _CCCL_HAS_NVBF16()
    if constexpr (is_integral_v<_Tp>)
    {
      return static_cast<double>(__x < __y ? __y : __x);
    }
    else
    {
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
#if _CCCL_HAS_FLOAT128()
        if constexpr (is_same_v<_Tp, __float128>)
        {
          NV_IF_TARGET(NV_PROVIDES_SM_100, (return ::__nv_fp128_fmax(__x, __y);))
        }
#endif // _CCCL_HAS_FLOAT128()
        if constexpr (is_same_v<_Tp, double>)
        {
          _CCCL_DISPATCH_SLOW_FP64_TARGET(return ::cuda::std::__fp64_fmax_fmin_int<true>(__x, __y);)
        }
#if _CCCL_USE_BUILTIN_FMAX()
        if constexpr (is_floating_point_v<_Tp>)
        {
// GCC builtins do not treat NaN properly
#  if _CCCL_COMPILER(GCC)
          NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::std::__with_builtin_fmax(__x, __y);))
#  else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC)
          return ::cuda::std::__with_builtin_fmax(__x, __y);
#  endif // !_CCCL_COMPILER(GCC)
        }
#endif // _CCCL_USE_BUILTIN_FMAX
      }
      if (::cuda::std::isnan(__x))
      {
        return __y;
      }
      else if (::cuda::std::isnan(__y))
      {
        return __x;
      }
      else
      {
        return __x < __y ? __y : __x;
      }
    }
}

[[nodiscard]] _CCCL_API constexpr float fmaxf(float __x, float __y) noexcept
{
  return ::cuda::std::fmax(__x, __y);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr long double fmaxl(long double __x, long double __y) noexcept
{
  return ::cuda::std::fmax(__x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(::cuda::is_floating_point_v<_Tp> _CCCL_AND ::cuda::is_floating_point_v<_Up>)
[[nodiscard]] _CCCL_API constexpr auto fmax(_Tp __x, _Up __y) noexcept
{
  using __result_type = __promote_t<_Tp, _Up>;
  static_assert(!(is_same_v<_Tp, __result_type> && is_same_v<_Up, __result_type>) );
  return ::cuda::std::fmax(static_cast<__result_type>(__x), static_cast<__result_type>(__y));
}

/***********************************************************************************************************************
 * fmin
 **********************************************************************************************************************/

// We do explicitly also enable GCC here, because that makes the condition below simpler
#if _CCCL_CHECK_BUILTIN(builtin_fmin) || _CCCL_COMPILER(GCC)
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_API _Tp __with_builtin_fmin(_Tp __x, _Tp __y) noexcept
{
#  if _CCCL_COMPILER(CLANG) && _CCCL_HOST_ARCH(ARM64) && _CCCL_HOST_COMPILATION()
  // Clang lowers fmin to AArch64 fminnm, which returns quiet NaN for signaling NaNs,
  // See https://github.com/llvm/llvm-project/issues/176624
  if (::cuda::std::isnan(__x))
  {
    return __y;
  }
  if (::cuda::std::isnan(__y))
  {
    return __x;
  }
#  endif // _CCCL_COMPILER(CLANG) && _CCCL_HOST_ARCH(ARM64) && _CCCL_HOST_COMPILATION()
  if constexpr (is_same_v<_Tp, float>)
  {
    return __builtin_fminf(__x, __y);
  }
  else if constexpr (is_same_v<_Tp, double>)
  {
    return __builtin_fmin(__x, __y);
  }
#  if _CCCL_HAS_LONG_DOUBLE()
  else if constexpr (is_same_v<_Tp, long double>)
  {
    return __builtin_fminl(__x, __y);
  }
#  endif // _CCCL_HAS_LONG_DOUBLE()
}

#  define _CCCL_USE_BUILTIN_FMIN() 1
#else //
#  define _CCCL_USE_BUILTIN_FMIN() 0
#endif // _CCCL_BUILTIN_FABSF

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr conditional_t<is_integral_v<_Tp>, double, _Tp> fmin(_Tp __x, _Tp __y) noexcept
{
#if _CCCL_HAS_NVFP16()
  // The half and bfloat16 branches can become identical under some CUDA Toolkit versions.
  // NOLINTBEGIN(bugprone-branch-clone)
  if constexpr (is_same_v<_Tp, ::__half>)
  {
#  if _CCCL_CTK_AT_LEAST(12, 2)
    return ::__hmin(__x, __y);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return ::__hmin(__x, __y);),
                      (return ::__float2half(::cuda::std::fmin(::__half2float(__x), ::__half2float(__y)));))
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
  }
  // NOLINTEND(bugprone-branch-clone)
  else
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
    if constexpr (is_same_v<_Tp, ::__nv_bfloat16>)
  {
#  if _CCCL_CTK_AT_LEAST(12, 2)
    return ::__hmin(__x, __y);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                      (return ::__hmin(__x, __y);),
                      (return ::__float2bfloat16(::cuda::std::fmin(::__bfloat162float(__x), ::__bfloat162float(__y)));))
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
  }
  else
#endif // _CCCL_HAS_NVBF16()
    if constexpr (is_integral_v<_Tp>)
    {
      return static_cast<double>(__y < __x ? __y : __x);
    }
    else
    {
      _CCCL_IF_NOT_CONSTEVAL_DEFAULT
      {
#if _CCCL_HAS_FLOAT128()
        if constexpr (is_same_v<_Tp, __float128>)
        {
          NV_IF_TARGET(NV_PROVIDES_SM_100, (return ::__nv_fp128_fmin(__x, __y);))
        }
#endif // _CCCL_HAS_FLOAT128()
        if constexpr (is_same_v<_Tp, double>)
        {
          _CCCL_DISPATCH_SLOW_FP64_TARGET(return ::cuda::std::__fp64_fmax_fmin_int<false>(__x, __y);)
        }
#if _CCCL_USE_BUILTIN_FMIN()
        if constexpr (is_floating_point_v<_Tp>)
        {
          // GCC builtins do not treat NaN properly
#  if _CCCL_COMPILER(GCC)
          NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::std::__with_builtin_fmin(__x, __y);))
#  else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC)
          return ::cuda::std::__with_builtin_fmin(__x, __y);
#  endif // !_CCCL_COMPILER(GCC)
        }
#endif // _CCCL_USE_BUILTIN_FMIN
      }
      if (::cuda::std::isnan(__x))
      {
        return __y;
      }
      else if (::cuda::std::isnan(__y))
      {
        return __x;
      }
      return __y < __x ? __y : __x;
    }
}

[[nodiscard]] _CCCL_API constexpr float fminf(float __x, float __y) noexcept
{
  return ::cuda::std::fmin(__x, __y);
}

#if _CCCL_HAS_LONG_DOUBLE()
[[nodiscard]] _CCCL_API constexpr long double fminl(long double __x, long double __y) noexcept
{
  return ::cuda::std::fmin(__x, __y);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(::cuda::is_floating_point_v<_Tp> _CCCL_AND ::cuda::is_floating_point_v<_Up>)
[[nodiscard]] _CCCL_API constexpr auto fmin(_Tp __x, _Up __y) noexcept
{
  using __result_type = __promote_t<_Tp, _Up>;
  static_assert(!(is_same_v<_Tp, __result_type> && is_same_v<_Up, __result_type>) );
  return ::cuda::std::fmin(static_cast<__result_type>(__x), static_cast<__result_type>(__y));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_MIN_MAX_H
