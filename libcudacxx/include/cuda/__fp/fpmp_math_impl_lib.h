//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_LIB_H
#define _CUDA___FP_FPMP_MATH_IMPL_LIB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_lib.h - fpmp2 math declarations for library (precompiled-kernel) mode
    ==================================================================================================
    Active only when _CCCL_FPMP_USE_LIB is defined. Declares the low-level
    fp32mp2/fp64mp2 kernels (the __fp32mp2_ and __fp64mp2_ entry points) provided
    by the compiled fpmp library together with the generic __fpmp2_ template
    wrappers (and their float/double specializations) that forward to those
    kernels. In header-only mode this file expands to nothing -- the kernels are
    provided by the per-family fpmp_math_impl_<family>.h headers instead.

    Included (under _CCCL_FPMP_USE_LIB) by <cuda/__fp/fpmp_math.h>.
*/

#include <cuda/__fp/fpmp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if (defined _CCCL_FPMP_USE_LIB)
/*
 * ============================================================================
 * Forwarding factory macros
 * ============================================================================
 * Every math function in library mode needs the same five declarations: the two
 * fp32mp2/fp64mp2 kernels the compiled library exports, the generic __fpmp2_
 * template, and its float and double specializations forwarding to those
 * kernels. The macros below emit that group from the function name, so the
 * list further down states what the library provides and nothing else.
 *
 * A layout suffix names the signature, matching the dispatch macros in
 * fpmp_math_impl.h: 1A / 2A / 3A / 4A count the (hi, lo) input pairs, RET* gives
 * a return type other than a pair, FP_INT and FP_LINT append an integer
 * exponent, INT_FP puts an order argument first, and SINCOS / MODF / FREXP / QUO
 * name the functions with a second output.
 * ============================================================================
 */

// The parameter lists stay one per line: clang-format cannot tell that T is a type
// here and rewrites `T* p` into `T *p`.
// clang-format off
#  define _CCCL_FPMP_LIB_PARAMS_1A(T) const T __x_hi, const T __x_lo, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_2A(T) const T __x_hi, const T __x_lo, const T __y_hi, const T __y_lo, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_2A_YX(T) const T __y_hi, const T __y_lo, const T __x_hi, const T __x_lo, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_3A(T) const T __a_hi, const T __a_lo, const T __b_hi, const T __b_lo, const T __c_hi, const T __c_lo, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_4A(T) const T __a_hi, const T __a_lo, const T __b_hi, const T __b_lo, const T __c_hi, const T __c_lo, const T __d_hi, const T __d_lo, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_1A_RET(T) const T __x_hi, const T __x_lo
#  define _CCCL_FPMP_LIB_PARAMS_FP_INT(T) const T __x_hi, const T __x_lo, int __n, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_FP_LINT(T) const T __x_hi, const T __x_lo, long int __n, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_INT_FP(T) int __n, const T __x_hi, const T __x_lo, T* __res_hi, T* __res_lo
#  define _CCCL_FPMP_LIB_PARAMS_1A_SINCOS(T) const T __x_hi, const T __x_lo, T* __sin_hi, T* __sin_lo, T* __cos_hi, T* __cos_lo
#  define _CCCL_FPMP_LIB_PARAMS_1A_MODF(T) const T __x_hi, const T __x_lo, T* __res_hi, T* __res_lo, T* __iptr_hi, T* __iptr_lo
#  define _CCCL_FPMP_LIB_PARAMS_1A_FREXP(T) const T __x_hi, const T __x_lo, T* __res_hi, T* __res_lo, int* __nptr
#  define _CCCL_FPMP_LIB_PARAMS_2A_QUO(T) const T __x_hi, const T __x_lo, const T __y_hi, const T __y_lo, T* __res_hi, T* __res_lo, int* __quo

#  define _CCCL_FPMP_LIB_ARGS_1A __x_hi, __x_lo, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_2A __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_2A_YX __y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_3A __a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_4A __a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_1A_RET __x_hi, __x_lo
#  define _CCCL_FPMP_LIB_ARGS_FP_INT __x_hi, __x_lo, __n, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_FP_LINT __x_hi, __x_lo, __n, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_INT_FP __n, __x_hi, __x_lo, __res_hi, __res_lo
#  define _CCCL_FPMP_LIB_ARGS_1A_SINCOS __x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo
#  define _CCCL_FPMP_LIB_ARGS_1A_MODF __x_hi, __x_lo, __res_hi, __res_lo, __iptr_hi, __iptr_lo
#  define _CCCL_FPMP_LIB_ARGS_1A_FREXP __x_hi, __x_lo, __res_hi, __res_lo, __nptr
#  define _CCCL_FPMP_LIB_ARGS_2A_QUO __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo
// clang-format on

#  define _CCCL_FPMP_LIB_VOID(name, layout)                                                                   \
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_##name(_CCCL_FPMP_LIB_PARAMS_##layout(float)) noexcept;            \
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_##name(_CCCL_FPMP_LIB_PARAMS_##layout(double)) noexcept;           \
    template <typename _Tp>                                                                                   \
    _CCCL_HOST_DEVICE_API inline void __fpmp2_##name(_CCCL_FPMP_LIB_PARAMS_##layout(_Tp)) noexcept;           \
    template <>                                                                                               \
    _CCCL_HOST_DEVICE_API inline void __fpmp2_##name<float>(_CCCL_FPMP_LIB_PARAMS_##layout(float)) noexcept   \
    {                                                                                                         \
      __fp32mp2_##name(_CCCL_FPMP_LIB_ARGS_##layout);                                                         \
    }                                                                                                         \
    template <>                                                                                               \
    _CCCL_HOST_DEVICE_API inline void __fpmp2_##name<double>(_CCCL_FPMP_LIB_PARAMS_##layout(double)) noexcept \
    {                                                                                                         \
      __fp64mp2_##name(_CCCL_FPMP_LIB_ARGS_##layout);                                                         \
    }

// icdf takes a uniform integer of the given width instead of a limb pair and
// exists for fp32mp2 only, so it has no generic template and no fp64mp2 kernel.
// The wrapper is an overload rather than a specialization, selected by the
// argument type.
#  define _CCCL_FPMP_LIB_UINT_FP(name, width)                                                                        \
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_##name##width(                                                            \
      uint##width##_t __x, float* __res_hi, float* __res_lo) noexcept;                                               \
    _CCCL_HOST_DEVICE_API inline void __fpmp2_##name(uint##width##_t __x, float* __res_hi, float* __res_lo) noexcept \
    {                                                                                                                \
      __fp32mp2_##name##width(__x, __res_hi, __res_lo);                                                              \
    }

#  define _CCCL_FPMP_LIB_RET(ret, name, layout)                                                              \
    _CCCL_FPMP_BUILTIN_DECL ret __fp32mp2_##name(_CCCL_FPMP_LIB_PARAMS_##layout(float)) noexcept;            \
    _CCCL_FPMP_BUILTIN_DECL ret __fp64mp2_##name(_CCCL_FPMP_LIB_PARAMS_##layout(double)) noexcept;           \
    template <typename _Tp>                                                                                  \
    _CCCL_HOST_DEVICE_API inline ret __fpmp2_##name(_CCCL_FPMP_LIB_PARAMS_##layout(_Tp)) noexcept;           \
    template <>                                                                                              \
    _CCCL_HOST_DEVICE_API inline ret __fpmp2_##name<float>(_CCCL_FPMP_LIB_PARAMS_##layout(float)) noexcept   \
    {                                                                                                        \
      return __fp32mp2_##name(_CCCL_FPMP_LIB_ARGS_##layout);                                                 \
    }                                                                                                        \
    template <>                                                                                              \
    _CCCL_HOST_DEVICE_API inline ret __fpmp2_##name<double>(_CCCL_FPMP_LIB_PARAMS_##layout(double)) noexcept \
    {                                                                                                        \
      return __fp64mp2_##name(_CCCL_FPMP_LIB_ARGS_##layout);                                                 \
    }

#  define _CCCL_FPMP_LIB_1A(name)        _CCCL_FPMP_LIB_VOID(name, 1A)
#  define _CCCL_FPMP_LIB_2A(name)        _CCCL_FPMP_LIB_VOID(name, 2A)
#  define _CCCL_FPMP_LIB_2A_YX(name)     _CCCL_FPMP_LIB_VOID(name, 2A_YX)
#  define _CCCL_FPMP_LIB_3A(name)        _CCCL_FPMP_LIB_VOID(name, 3A)
#  define _CCCL_FPMP_LIB_4A(name)        _CCCL_FPMP_LIB_VOID(name, 4A)
#  define _CCCL_FPMP_LIB_FP_INT(name)    _CCCL_FPMP_LIB_VOID(name, FP_INT)
#  define _CCCL_FPMP_LIB_FP_LINT(name)   _CCCL_FPMP_LIB_VOID(name, FP_LINT)
#  define _CCCL_FPMP_LIB_INT_FP(name)    _CCCL_FPMP_LIB_VOID(name, INT_FP)
#  define _CCCL_FPMP_LIB_1A_SINCOS(name) _CCCL_FPMP_LIB_VOID(name, 1A_SINCOS)
#  define _CCCL_FPMP_LIB_1A_MODF(name)   _CCCL_FPMP_LIB_VOID(name, 1A_MODF)
#  define _CCCL_FPMP_LIB_1A_FREXP(name)  _CCCL_FPMP_LIB_VOID(name, 1A_FREXP)
#  define _CCCL_FPMP_LIB_2A_QUO(name)    _CCCL_FPMP_LIB_VOID(name, 2A_QUO)
#  define _CCCL_FPMP_LIB_1A_RETINT(name) _CCCL_FPMP_LIB_RET(int, name, 1A_RET)
#  define _CCCL_FPMP_LIB_1A_RETLL(name)  _CCCL_FPMP_LIB_RET(long long int, name, 1A_RET)
#  define _CCCL_FPMP_LIB_1A_RETL(name)   _CCCL_FPMP_LIB_RET(long int, name, 1A_RET)

/*
 * ============================================================================
 * Math functions provided by the compiled library
 * ============================================================================
 */

_CCCL_FPMP_LIB_1A(exp)
_CCCL_FPMP_LIB_1A(log)
_CCCL_FPMP_LIB_1A(log2)
_CCCL_FPMP_LIB_1A(log10)
_CCCL_FPMP_LIB_1A(log1p)
_CCCL_FPMP_LIB_2A(pow)
_CCCL_FPMP_LIB_1A(cbrt)
_CCCL_FPMP_LIB_1A(sin)
_CCCL_FPMP_LIB_1A(cos)
_CCCL_FPMP_LIB_1A_SINCOS(sincos)
_CCCL_FPMP_LIB_1A(asin)
_CCCL_FPMP_LIB_1A(acos)
_CCCL_FPMP_LIB_1A(atan)
_CCCL_FPMP_LIB_2A_YX(atan2)
_CCCL_FPMP_LIB_1A(sinh)
_CCCL_FPMP_LIB_1A(cosh)
_CCCL_FPMP_LIB_1A(tanh)
_CCCL_FPMP_LIB_1A(erf)
_CCCL_FPMP_LIB_1A(erfc)
_CCCL_FPMP_LIB_1A(normcdfinv)
_CCCL_FPMP_LIB_1A(acosh)
_CCCL_FPMP_LIB_1A(asinh)
_CCCL_FPMP_LIB_1A(atanh)
_CCCL_FPMP_LIB_1A(tan)
_CCCL_FPMP_LIB_1A(exp2)
_CCCL_FPMP_LIB_1A(exp10)
_CCCL_FPMP_LIB_1A(expm1)
_CCCL_FPMP_LIB_1A(logb)
_CCCL_FPMP_LIB_1A(ceil)
_CCCL_FPMP_LIB_1A(floor)
_CCCL_FPMP_LIB_1A(trunc)
_CCCL_FPMP_LIB_1A(round)
_CCCL_FPMP_LIB_1A(rint)
_CCCL_FPMP_LIB_1A(nearbyint)
_CCCL_FPMP_LIB_1A(fabs)
_CCCL_FPMP_LIB_1A(lgamma)
_CCCL_FPMP_LIB_1A(tgamma)
_CCCL_FPMP_LIB_1A(j0)
_CCCL_FPMP_LIB_1A(j1)
_CCCL_FPMP_LIB_1A(y0)
_CCCL_FPMP_LIB_1A(y1)
_CCCL_FPMP_LIB_1A(cyl_bessel_i0)
_CCCL_FPMP_LIB_1A(cyl_bessel_i1)
_CCCL_FPMP_LIB_1A(sinpi)
_CCCL_FPMP_LIB_1A(cospi)
_CCCL_FPMP_LIB_1A(normcdf)
_CCCL_FPMP_LIB_1A(rcbrt)
_CCCL_FPMP_LIB_1A(erfcinv)
_CCCL_FPMP_LIB_1A(erfinv)
_CCCL_FPMP_LIB_1A(erfcx)
_CCCL_FPMP_LIB_1A(boys_f0)
_CCCL_FPMP_LIB_3A(norm3d)
_CCCL_FPMP_LIB_4A(norm4d)
_CCCL_FPMP_LIB_3A(rnorm3d)
_CCCL_FPMP_LIB_4A(rnorm4d)
_CCCL_FPMP_LIB_2A(fmax)
_CCCL_FPMP_LIB_2A(fmin)
_CCCL_FPMP_LIB_2A(max)
_CCCL_FPMP_LIB_2A(min)
_CCCL_FPMP_LIB_2A(fmod)
_CCCL_FPMP_LIB_2A(remainder)
_CCCL_FPMP_LIB_2A(hypot)
_CCCL_FPMP_LIB_2A(copysign)
_CCCL_FPMP_LIB_2A(fdim)
_CCCL_FPMP_LIB_2A(nextafter)
_CCCL_FPMP_LIB_2A(rhypot)
_CCCL_FPMP_LIB_2A_QUO(remquo)
_CCCL_FPMP_LIB_1A_RETINT(ilogb)
_CCCL_FPMP_LIB_1A_RETLL(llrint)
_CCCL_FPMP_LIB_1A_RETLL(llround)
_CCCL_FPMP_LIB_1A_RETL(lrint)
_CCCL_FPMP_LIB_1A_RETL(lround)
_CCCL_FPMP_LIB_1A_RETINT(isfinite)
_CCCL_FPMP_LIB_1A_RETINT(isinf)
_CCCL_FPMP_LIB_1A_RETINT(isnan)
_CCCL_FPMP_LIB_1A_RETINT(signbit)
_CCCL_FPMP_LIB_FP_INT(ldexp)
_CCCL_FPMP_LIB_FP_INT(scalbn)
_CCCL_FPMP_LIB_FP_LINT(scalbln)
_CCCL_FPMP_LIB_INT_FP(jn)
_CCCL_FPMP_LIB_INT_FP(yn)
_CCCL_FPMP_LIB_1A_FREXP(frexp)
_CCCL_FPMP_LIB_1A_MODF(modf)
_CCCL_FPMP_LIB_1A_SINCOS(sincospi)

_CCCL_FPMP_LIB_UINT_FP(icdf, 32)
_CCCL_FPMP_LIB_UINT_FP(icdf, 64)

/* Keep the factory macros out of downstream translation units. */
#  undef _CCCL_FPMP_LIB_PARAMS_1A
#  undef _CCCL_FPMP_LIB_PARAMS_2A
#  undef _CCCL_FPMP_LIB_PARAMS_2A_YX
#  undef _CCCL_FPMP_LIB_PARAMS_3A
#  undef _CCCL_FPMP_LIB_PARAMS_4A
#  undef _CCCL_FPMP_LIB_PARAMS_1A_RET
#  undef _CCCL_FPMP_LIB_PARAMS_FP_INT
#  undef _CCCL_FPMP_LIB_PARAMS_FP_LINT
#  undef _CCCL_FPMP_LIB_PARAMS_INT_FP
#  undef _CCCL_FPMP_LIB_PARAMS_1A_SINCOS
#  undef _CCCL_FPMP_LIB_PARAMS_1A_MODF
#  undef _CCCL_FPMP_LIB_PARAMS_1A_FREXP
#  undef _CCCL_FPMP_LIB_PARAMS_2A_QUO
#  undef _CCCL_FPMP_LIB_ARGS_1A
#  undef _CCCL_FPMP_LIB_ARGS_2A
#  undef _CCCL_FPMP_LIB_ARGS_2A_YX
#  undef _CCCL_FPMP_LIB_ARGS_3A
#  undef _CCCL_FPMP_LIB_ARGS_4A
#  undef _CCCL_FPMP_LIB_ARGS_1A_RET
#  undef _CCCL_FPMP_LIB_ARGS_FP_INT
#  undef _CCCL_FPMP_LIB_ARGS_FP_LINT
#  undef _CCCL_FPMP_LIB_ARGS_INT_FP
#  undef _CCCL_FPMP_LIB_ARGS_1A_SINCOS
#  undef _CCCL_FPMP_LIB_ARGS_1A_MODF
#  undef _CCCL_FPMP_LIB_ARGS_1A_FREXP
#  undef _CCCL_FPMP_LIB_ARGS_2A_QUO
#  undef _CCCL_FPMP_LIB_UINT_FP
#  undef _CCCL_FPMP_LIB_VOID
#  undef _CCCL_FPMP_LIB_RET
#  undef _CCCL_FPMP_LIB_1A
#  undef _CCCL_FPMP_LIB_2A
#  undef _CCCL_FPMP_LIB_2A_YX
#  undef _CCCL_FPMP_LIB_3A
#  undef _CCCL_FPMP_LIB_4A
#  undef _CCCL_FPMP_LIB_FP_INT
#  undef _CCCL_FPMP_LIB_FP_LINT
#  undef _CCCL_FPMP_LIB_INT_FP
#  undef _CCCL_FPMP_LIB_1A_SINCOS
#  undef _CCCL_FPMP_LIB_1A_MODF
#  undef _CCCL_FPMP_LIB_1A_FREXP
#  undef _CCCL_FPMP_LIB_2A_QUO
#  undef _CCCL_FPMP_LIB_1A_RETINT
#  undef _CCCL_FPMP_LIB_1A_RETLL
#  undef _CCCL_FPMP_LIB_1A_RETL
#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_LIB_H
