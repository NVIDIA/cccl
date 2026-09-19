//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_STAT_MATH_H
#define _CUDA___FP_FPTOOL_STAT_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fptool_stat_math.h
//! @brief Math API for `fpmp2_stat`
//!
//! Mirrors the free functions of `<cuda/fpmp_math>` for the statistics-collecting
//! wrapper: each one unwraps its arguments, calls the `fpmp2` implementation and wraps
//! the result again, so results are bit-identical to the plain type. Including this
//! header lets unqualified calls such as `exp(x)` or `hypot(x, y)` resolve for a `_stat`
//! type without touching application sources.
//!
//! The wrappers exist because overload resolution never applies a user-defined
//! conversion while deducing template arguments: `exp(x)` on a `_stat` value would
//! otherwise not find `exp(const fpmp2&)` at all.
//!
//! None of these functions is instrumented. They are composites, and counting the
//! operations inside them would swamp the counters of the algorithm under study; the
//! counters cover the binary operators only, see `<cuda/__fp/fptool_stat.h>`.
//!
//! `icdf` needs no wrapper: it takes an integer rather than a floating-point value, so
//! its `fpmp2` result converts implicitly, as in `fp32mp2_stat v = icdf(bits);`.

#include <cuda/__fp/fpmp_math.h>
#include <cuda/__fp/fptool_stat.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// Each wrapper delegates to the fpmp2 overload of the same name: an unqualified call
// with an fpmp2 argument cannot select the _stat overload, so there is no recursion.
#define _CCCL_FPMP_STAT_MATH_UNARY(_Name)                                         \
  template <class _FpType, fpmp2_accuracy _TypeAcc>                               \
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> _Name( \
    const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept                            \
  {                                                                               \
    return fpmp2_stat<_FpType, _TypeAcc>(_Name(__x.as_fpmp2()));                  \
  }

#define _CCCL_FPMP_STAT_MATH_BINARY(_Name)                                                       \
  template <class _FpType, fpmp2_accuracy _TypeAcc>                                              \
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> _Name(                \
    const fpmp2_stat<_FpType, _TypeAcc>& __x, const fpmp2_stat<_FpType, _TypeAcc>& __y) noexcept \
  {                                                                                              \
    return fpmp2_stat<_FpType, _TypeAcc>(_Name(__x.as_fpmp2(), __y.as_fpmp2()));                 \
  }

#define _CCCL_FPMP_STAT_MATH_TERNARY(_Name)                                                      \
  template <class _FpType, fpmp2_accuracy _TypeAcc>                                              \
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> _Name(                \
    const fpmp2_stat<_FpType, _TypeAcc>& __a,                                                    \
    const fpmp2_stat<_FpType, _TypeAcc>& __b,                                                    \
    const fpmp2_stat<_FpType, _TypeAcc>& __c) noexcept                                           \
  {                                                                                              \
    return fpmp2_stat<_FpType, _TypeAcc>(_Name(__a.as_fpmp2(), __b.as_fpmp2(), __c.as_fpmp2())); \
  }

#define _CCCL_FPMP_STAT_MATH_QUATERNARY(_Name)                                                                   \
  template <class _FpType, fpmp2_accuracy _TypeAcc>                                                              \
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> _Name(                                \
    const fpmp2_stat<_FpType, _TypeAcc>& __a,                                                                    \
    const fpmp2_stat<_FpType, _TypeAcc>& __b,                                                                    \
    const fpmp2_stat<_FpType, _TypeAcc>& __c,                                                                    \
    const fpmp2_stat<_FpType, _TypeAcc>& __d) noexcept                                                           \
  {                                                                                                              \
    return fpmp2_stat<_FpType, _TypeAcc>(_Name(__a.as_fpmp2(), __b.as_fpmp2(), __c.as_fpmp2(), __d.as_fpmp2())); \
  }

#define _CCCL_FPMP_STAT_MATH_UNARY_RET(_Ret, _Name)                                                        \
  template <class _FpType, fpmp2_accuracy _TypeAcc>                                                        \
  [[nodiscard]] _CCCL_HOST_DEVICE_API inline _Ret _Name(const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept \
  {                                                                                                        \
    return _Name(__x.as_fpmp2());                                                                          \
  }

// Exponential, logarithmic and power functions
_CCCL_FPMP_STAT_MATH_UNARY(exp)
_CCCL_FPMP_STAT_MATH_UNARY(exp2)
_CCCL_FPMP_STAT_MATH_UNARY(exp10)
_CCCL_FPMP_STAT_MATH_UNARY(expm1)
_CCCL_FPMP_STAT_MATH_UNARY(log)
_CCCL_FPMP_STAT_MATH_UNARY(log2)
_CCCL_FPMP_STAT_MATH_UNARY(log10)
_CCCL_FPMP_STAT_MATH_UNARY(log1p)
_CCCL_FPMP_STAT_MATH_UNARY(logb)
_CCCL_FPMP_STAT_MATH_UNARY(cbrt)
_CCCL_FPMP_STAT_MATH_UNARY(rcbrt)
_CCCL_FPMP_STAT_MATH_BINARY(pow)

// Trigonometric and hyperbolic functions
_CCCL_FPMP_STAT_MATH_UNARY(sin)
_CCCL_FPMP_STAT_MATH_UNARY(cos)
_CCCL_FPMP_STAT_MATH_UNARY(tan)
_CCCL_FPMP_STAT_MATH_UNARY(sinpi)
_CCCL_FPMP_STAT_MATH_UNARY(cospi)
_CCCL_FPMP_STAT_MATH_UNARY(asin)
_CCCL_FPMP_STAT_MATH_UNARY(acos)
_CCCL_FPMP_STAT_MATH_UNARY(atan)
_CCCL_FPMP_STAT_MATH_BINARY(atan2)
_CCCL_FPMP_STAT_MATH_UNARY(sinh)
_CCCL_FPMP_STAT_MATH_UNARY(cosh)
_CCCL_FPMP_STAT_MATH_UNARY(tanh)
_CCCL_FPMP_STAT_MATH_UNARY(asinh)
_CCCL_FPMP_STAT_MATH_UNARY(acosh)
_CCCL_FPMP_STAT_MATH_UNARY(atanh)

// Error, distribution and gamma functions
_CCCL_FPMP_STAT_MATH_UNARY(erf)
_CCCL_FPMP_STAT_MATH_UNARY(erfc)
_CCCL_FPMP_STAT_MATH_UNARY(erfinv)
_CCCL_FPMP_STAT_MATH_UNARY(erfcinv)
_CCCL_FPMP_STAT_MATH_UNARY(erfcx)
_CCCL_FPMP_STAT_MATH_UNARY(normcdf)
_CCCL_FPMP_STAT_MATH_UNARY(normcdfinv)
_CCCL_FPMP_STAT_MATH_UNARY(boys_f0)
_CCCL_FPMP_STAT_MATH_UNARY(lgamma)
_CCCL_FPMP_STAT_MATH_UNARY(tgamma)

// Bessel functions
_CCCL_FPMP_STAT_MATH_UNARY(j0)
_CCCL_FPMP_STAT_MATH_UNARY(j1)
_CCCL_FPMP_STAT_MATH_UNARY(y0)
_CCCL_FPMP_STAT_MATH_UNARY(y1)
_CCCL_FPMP_STAT_MATH_UNARY(cyl_bessel_i0)
_CCCL_FPMP_STAT_MATH_UNARY(cyl_bessel_i1)

// Nearest-integer and sign functions
_CCCL_FPMP_STAT_MATH_UNARY(ceil)
_CCCL_FPMP_STAT_MATH_UNARY(floor)
_CCCL_FPMP_STAT_MATH_UNARY(trunc)
_CCCL_FPMP_STAT_MATH_UNARY(round)
_CCCL_FPMP_STAT_MATH_UNARY(rint)
_CCCL_FPMP_STAT_MATH_UNARY(nearbyint)
_CCCL_FPMP_STAT_MATH_UNARY(fabs)

// Minimum, maximum, remainder and other binary functions
_CCCL_FPMP_STAT_MATH_BINARY(fmax)
_CCCL_FPMP_STAT_MATH_BINARY(fmin)
_CCCL_FPMP_STAT_MATH_BINARY(max)
_CCCL_FPMP_STAT_MATH_BINARY(min)
_CCCL_FPMP_STAT_MATH_BINARY(fmod)
_CCCL_FPMP_STAT_MATH_BINARY(remainder)
_CCCL_FPMP_STAT_MATH_BINARY(hypot)
_CCCL_FPMP_STAT_MATH_BINARY(rhypot)
_CCCL_FPMP_STAT_MATH_BINARY(copysign)
_CCCL_FPMP_STAT_MATH_BINARY(fdim)
_CCCL_FPMP_STAT_MATH_BINARY(nextafter)

// Norms
_CCCL_FPMP_STAT_MATH_TERNARY(norm3d)
_CCCL_FPMP_STAT_MATH_TERNARY(rnorm3d)
_CCCL_FPMP_STAT_MATH_QUATERNARY(norm4d)
_CCCL_FPMP_STAT_MATH_QUATERNARY(rnorm4d)

// Integer-returning and classification functions
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, ilogb)
_CCCL_FPMP_STAT_MATH_UNARY_RET(long long int, llrint)
_CCCL_FPMP_STAT_MATH_UNARY_RET(long long int, llround)
_CCCL_FPMP_STAT_MATH_UNARY_RET(long int, lrint)
_CCCL_FPMP_STAT_MATH_UNARY_RET(long int, lround)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, fpmp_isfinite)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, fpmp_isinf)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, fpmp_isnan)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, fpmp_signbit)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, isfinite)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, isinf)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, isnan)
_CCCL_FPMP_STAT_MATH_UNARY_RET(int, signbit)

#undef _CCCL_FPMP_STAT_MATH_UNARY
#undef _CCCL_FPMP_STAT_MATH_BINARY
#undef _CCCL_FPMP_STAT_MATH_TERNARY
#undef _CCCL_FPMP_STAT_MATH_QUATERNARY
#undef _CCCL_FPMP_STAT_MATH_UNARY_RET

// Functions with out-pointers or plain integer arguments, which the macros above do not
// cover.

//! @brief Sine and cosine of the same argument
template <class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline void
sincos(const fpmp2_stat<_FpType, _TypeAcc>& __x,
       fpmp2_stat<_FpType, _TypeAcc>* __s,
       fpmp2_stat<_FpType, _TypeAcc>* __c) noexcept
{
  fpmp2<_FpType, _TypeAcc> __sin_res;
  fpmp2<_FpType, _TypeAcc> __cos_res;
  sincos(__x.as_fpmp2(), &__sin_res, &__cos_res);
  *__s = fpmp2_stat<_FpType, _TypeAcc>(__sin_res);
  *__c = fpmp2_stat<_FpType, _TypeAcc>(__cos_res);
}

//! @brief Sine and cosine of pi times the argument
template <class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API inline void
sincospi(const fpmp2_stat<_FpType, _TypeAcc>& __x,
         fpmp2_stat<_FpType, _TypeAcc>* __s,
         fpmp2_stat<_FpType, _TypeAcc>* __c) noexcept
{
  fpmp2<_FpType, _TypeAcc> __sin_res;
  fpmp2<_FpType, _TypeAcc> __cos_res;
  sincospi(__x.as_fpmp2(), &__sin_res, &__cos_res);
  *__s = fpmp2_stat<_FpType, _TypeAcc>(__sin_res);
  *__c = fpmp2_stat<_FpType, _TypeAcc>(__cos_res);
}

//! @brief Remainder of the division, together with part of the quotient
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
remquo(const fpmp2_stat<_FpType, _TypeAcc>& __x, const fpmp2_stat<_FpType, _TypeAcc>& __y, int* __quo) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(remquo(__x.as_fpmp2(), __y.as_fpmp2(), __quo));
}

//! @brief Multiply by a power of two
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
ldexp(const fpmp2_stat<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(ldexp(__x.as_fpmp2(), __n));
}

//! @brief Multiply by a power of the radix
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
scalbn(const fpmp2_stat<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(scalbn(__x.as_fpmp2(), __n));
}

//! @brief Multiply by a power of the radix, with a `long` exponent
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
scalbln(const fpmp2_stat<_FpType, _TypeAcc>& __x, long int __n) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(scalbln(__x.as_fpmp2(), __n));
}

//! @brief Split into a normalized fraction and a power of two
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
frexp(const fpmp2_stat<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(frexp(__x.as_fpmp2(), __nptr));
}

//! @brief Split into integral and fractional parts
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
modf(const fpmp2_stat<_FpType, _TypeAcc>& __x, fpmp2_stat<_FpType, _TypeAcc>* __iptr) noexcept
{
  fpmp2<_FpType, _TypeAcc> __int_part;
  const fpmp2<_FpType, _TypeAcc> __frac_part = modf(__x.as_fpmp2(), &__int_part);
  *__iptr                                    = fpmp2_stat<_FpType, _TypeAcc>(__int_part);
  return fpmp2_stat<_FpType, _TypeAcc>(__frac_part);
}

//! @brief Bessel function of the first kind, order n
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
jn(int __n, const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(jn(__n, __x.as_fpmp2()));
}

//! @brief Bessel function of the second kind, order n
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
yn(int __n, const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(yn(__n, __x.as_fpmp2()));
}
} // namespace cuda::experimental

// ============================================================================
// cuda::std overloads for the standard <cmath> names, for the same reason as the fpmp2
// ones in fpmp_math.h: a qualified cuda::std::<fn>(x) call suppresses ADL, and without
// these it would narrow the value to double and compute a native-double result. Only
// names that cuda::std declares are provided.
// ============================================================================
_CCCL_BEGIN_NAMESPACE_CUDA_STD

#define _CCCL_FPMP_STAT_STD_UNARY(_Name)                                           \
  template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>          \
  _CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc> _Name( \
    const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept       \
  {                                                                                \
    return ::cuda::experimental::_Name(__x);                                       \
  }

_CCCL_FPMP_STAT_STD_UNARY(exp)
_CCCL_FPMP_STAT_STD_UNARY(exp2)
_CCCL_FPMP_STAT_STD_UNARY(expm1)
_CCCL_FPMP_STAT_STD_UNARY(log)
_CCCL_FPMP_STAT_STD_UNARY(log2)
_CCCL_FPMP_STAT_STD_UNARY(log10)
_CCCL_FPMP_STAT_STD_UNARY(log1p)
_CCCL_FPMP_STAT_STD_UNARY(logb)
_CCCL_FPMP_STAT_STD_UNARY(cbrt)
_CCCL_FPMP_STAT_STD_UNARY(sin)
_CCCL_FPMP_STAT_STD_UNARY(cos)
_CCCL_FPMP_STAT_STD_UNARY(tan)
_CCCL_FPMP_STAT_STD_UNARY(asin)
_CCCL_FPMP_STAT_STD_UNARY(acos)
_CCCL_FPMP_STAT_STD_UNARY(atan)
_CCCL_FPMP_STAT_STD_UNARY(sinh)
_CCCL_FPMP_STAT_STD_UNARY(cosh)
_CCCL_FPMP_STAT_STD_UNARY(tanh)
_CCCL_FPMP_STAT_STD_UNARY(asinh)
_CCCL_FPMP_STAT_STD_UNARY(acosh)
_CCCL_FPMP_STAT_STD_UNARY(atanh)
_CCCL_FPMP_STAT_STD_UNARY(erf)
_CCCL_FPMP_STAT_STD_UNARY(erfc)
_CCCL_FPMP_STAT_STD_UNARY(tgamma)
_CCCL_FPMP_STAT_STD_UNARY(lgamma)
_CCCL_FPMP_STAT_STD_UNARY(ceil)
_CCCL_FPMP_STAT_STD_UNARY(floor)
_CCCL_FPMP_STAT_STD_UNARY(trunc)
_CCCL_FPMP_STAT_STD_UNARY(round)
_CCCL_FPMP_STAT_STD_UNARY(rint)
_CCCL_FPMP_STAT_STD_UNARY(nearbyint)
_CCCL_FPMP_STAT_STD_UNARY(fabs)

#undef _CCCL_FPMP_STAT_STD_UNARY

#define _CCCL_FPMP_STAT_STD_BINARY(_Name)                                          \
  template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>          \
  _CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc> _Name( \
    const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x,                \
    const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __y) noexcept       \
  {                                                                                \
    return ::cuda::experimental::_Name(__x, __y);                                  \
  }

_CCCL_FPMP_STAT_STD_BINARY(pow)
_CCCL_FPMP_STAT_STD_BINARY(atan2)
_CCCL_FPMP_STAT_STD_BINARY(fmod)
_CCCL_FPMP_STAT_STD_BINARY(remainder)
_CCCL_FPMP_STAT_STD_BINARY(hypot)
_CCCL_FPMP_STAT_STD_BINARY(fmax)
_CCCL_FPMP_STAT_STD_BINARY(fmin)
_CCCL_FPMP_STAT_STD_BINARY(copysign)
_CCCL_FPMP_STAT_STD_BINARY(fdim)
_CCCL_FPMP_STAT_STD_BINARY(nextafter)

#undef _CCCL_FPMP_STAT_STD_BINARY

#define _CCCL_FPMP_STAT_STD_UNARY_RET(_Ret, _Name)                                                          \
  template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>                                   \
  _CCCL_HOST_DEVICE_API _Ret _Name(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept \
  {                                                                                                         \
    return ::cuda::experimental::_Name(__x);                                                                \
  }

_CCCL_FPMP_STAT_STD_UNARY_RET(int, ilogb)
_CCCL_FPMP_STAT_STD_UNARY_RET(long long int, llrint)
_CCCL_FPMP_STAT_STD_UNARY_RET(long long int, llround)
_CCCL_FPMP_STAT_STD_UNARY_RET(long int, lrint)
_CCCL_FPMP_STAT_STD_UNARY_RET(long int, lround)
_CCCL_FPMP_STAT_STD_UNARY_RET(int, isfinite)
_CCCL_FPMP_STAT_STD_UNARY_RET(int, isinf)
_CCCL_FPMP_STAT_STD_UNARY_RET(int, isnan)
_CCCL_FPMP_STAT_STD_UNARY_RET(int, signbit)

#undef _CCCL_FPMP_STAT_STD_UNARY_RET

// Functions with special signatures (extra scalar / out-pointer arguments).
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>
ldexp(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return ::cuda::experimental::ldexp(__x, __n);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>
scalbn(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x, int __n) noexcept
{
  return ::cuda::experimental::scalbn(__x, __n);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>
scalbln(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x, long int __n) noexcept
{
  return ::cuda::experimental::scalbln(__x, __n);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>
frexp(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{
  return ::cuda::experimental::frexp(__x, __nptr);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>
modf(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x,
     ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>* __iptr) noexcept
{
  return ::cuda::experimental::modf(__x, __iptr);
}
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
_CCCL_HOST_DEVICE_API ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>
remquo(const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __x,
       const ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>& __y,
       int* __quo) noexcept
{
  return ::cuda::experimental::remquo(__x, __y, __quo);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_STAT_MATH_H
