// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MATH_EXPONENTIAL_FUNCTIONS_H
#define _LIBCUDACXX___MATH_EXPONENTIAL_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>
#include <cuda/std/cstdint>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// exp

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float exp(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPF)
  return _CCCL_BUILTIN_EXPF(__x);
#else // ^^^ _CCCL_BUILTIN_EXPF ^^^ // vvv !_CCCL_BUILTIN_EXPF vvv
  return ::expf(__x);
#endif // !_CCCL_BUILTIN_EXPF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float expf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPF)
  return _CCCL_BUILTIN_EXPF(__x);
#else // ^^^ _CCCL_BUILTIN_EXPF ^^^ // vvv !_CCCL_BUILTIN_EXPF vvv
  return ::expf(__x);
#endif // !_CCCL_BUILTIN_EXPF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double exp(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP)
  return _CCCL_BUILTIN_EXP(__x);
#else // ^^^ _CCCL_BUILTIN_EXP ^^^ // vvv !_CCCL_BUILTIN_EXP vvv
  return ::exp(__x);
#endif // !_CCCL_BUILTIN_EXP
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double exp(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPL)
  return _CCCL_BUILTIN_EXPL(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPL ^^^ // vvv !_CCCL_BUILTIN_EXPL vvv
  return ::expl(__x);
#  endif // !_CCCL_BUILTIN_EXPL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double expl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPL)
  return _CCCL_BUILTIN_EXPL(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPL ^^^ // vvv !_CCCL_BUILTIN_EXPL vvv
  return ::expl(__x);
#  endif // !_CCCL_BUILTIN_EXPL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half exp(__half __x) noexcept
{
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (return ::hexp(__x);), ({
                        float __xf            = __half2float(__x);
                        __xf                  = ::expf(__xf);
                        __half_raw __ret_repr = ::__float2half_rn(__xf);

                        uint16_t __repr = _CUDA_VSTD::__fp_get_storage(__x);
                        switch (__repr)
                        {
                          case 8057:
                          case 9679:
                            __ret_repr.x -= 1;
                            break;

                          default:;
                        }

                        return __ret_repr;
                      }))
  }
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 exp(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hexp(__x);), (return __float2bfloat16(_CUDA_VSTD::expf(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double exp(_Integer __x) noexcept
{
  return _CUDA_VSTD::exp((double) __x);
}

// frexp

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float frexp(float __x, int* __e) noexcept
{
#if defined(_CCCL_BUILTIN_FREXPF)
  return _CCCL_BUILTIN_FREXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_FREXPF ^^^ // vvv !_CCCL_BUILTIN_FREXPF vvv
  return ::frexpf(__x, __e);
#endif // !_CCCL_BUILTIN_FREXPF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float frexpf(float __x, int* __e) noexcept
{
#if defined(_CCCL_BUILTIN_FREXPF)
  return _CCCL_BUILTIN_FREXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_FREXPF ^^^ // vvv !_CCCL_BUILTIN_FREXPF vvv
  return ::frexpf(__x, __e);
#endif // !_CCCL_BUILTIN_FREXPF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double frexp(double __x, int* __e) noexcept
{
#if defined(_CCCL_BUILTIN_FREXP)
  return _CCCL_BUILTIN_FREXP(__x, __e);
#else // ^^^ _CCCL_BUILTIN_FREXP ^^^ // vvv !_CCCL_BUILTIN_FREXP vvv
  return ::frexp(__x, __e);
#endif // !_CCCL_BUILTIN_FREXP
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double frexp(long double __x, int* __e) noexcept
{
#  if defined(_CCCL_BUILTIN_FREXPL)
  return _CCCL_BUILTIN_FREXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_FREXPL ^^^ // vvv !_CCCL_BUILTIN_FREXPL vvv
  return ::frexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_FREXPL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double frexpl(long double __x, int* __e) noexcept
{
#  if defined(_CCCL_BUILTIN_FREXPL)
  return _CCCL_BUILTIN_FREXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_FREXPL ^^^ // vvv !_CCCL_BUILTIN_FREXPL vvv
  return ::frexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_FREXPL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half frexp(__half __x, int* __e) noexcept
{
  return __float2half(_CUDA_VSTD::frexpf(__half2float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 frexp(__nv_bfloat16 __x, int* __e) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::frexpf(__bfloat162float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double frexp(_Integer __x, int* __e) noexcept
{
  return _CUDA_VSTD::frexp((double) __x, __e);
}

// ldexp

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float ldexp(float __x, int __e) noexcept
{
#if defined(_CCCL_BUILTIN_LDEXPF)
  return _CCCL_BUILTIN_LDEXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_LDEXPF ^^^ // vvv !_CCCL_BUILTIN_LDEXPF vvv
  return ::ldexpf(__x, __e);
#endif // !_CCCL_BUILTIN_LDEXPF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float ldexpf(float __x, int __e) noexcept
{
#if defined(_CCCL_BUILTIN_LDEXPF)
  return _CCCL_BUILTIN_LDEXPF(__x, __e);
#else // ^^^ _CCCL_BUILTIN_LDEXPF ^^^ // vvv !_CCCL_BUILTIN_LDEXPF vvv
  return ::ldexpf(__x, __e);
#endif // !_CCCL_BUILTIN_LDEXPF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double ldexp(double __x, int __e) noexcept
{
#if defined(_CCCL_BUILTIN_LDEXP)
  return _CCCL_BUILTIN_LDEXP(__x, __e);
#else // ^^^ _CCCL_BUILTIN_LDEXP ^^^ // vvv !_CCCL_BUILTIN_LDEXP vvv
  return ::ldexp(__x, __e);
#endif // !_CCCL_BUILTIN_LDEXP
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double ldexp(long double __x, int __e) noexcept
{
#  if defined(_CCCL_BUILTIN_LDEXPL)
  return _CCCL_BUILTIN_LDEXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_LDEXPL ^^^ // vvv !_CCCL_BUILTIN_LDEXPL vvv
  return ::ldexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_LDEXPL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double ldexpl(long double __x, int __e) noexcept
{
#  if defined(_CCCL_BUILTIN_LDEXPL)
  return _CCCL_BUILTIN_LDEXPL(__x, __e);
#  else // ^^^ _CCCL_BUILTIN_LDEXPL ^^^ // vvv !_CCCL_BUILTIN_LDEXPL vvv
  return ::ldexpl(__x, __e);
#  endif // !_CCCL_BUILTIN_LDEXPL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half ldexp(__half __x, int __e) noexcept
{
  return __float2half(_CUDA_VSTD::ldexpf(__half2float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 ldexp(__nv_bfloat16 __x, int __e) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::ldexpf(__bfloat162float(__x), __e));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double ldexp(_Integer __x, int __e) noexcept
{
  return _CUDA_VSTD::ldexp((double) __x, __e);
}

// exp2

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float exp2(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP2F)
  return _CCCL_BUILTIN_EXP2F(__x);
#else // ^^^ _CCCL_BUILTIN_EXP2F ^^^ // vvv !_CCCL_BUILTIN_EXP2F vvv
  return ::exp2f(__x);
#endif // !_CCCL_BUILTIN_EXP2F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float exp2f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP2F)
  return _CCCL_BUILTIN_EXP2F(__x);
#else // ^^^ _CCCL_BUILTIN_EXP2F ^^^ // vvv !_CCCL_BUILTIN_EXP2F vvv
  return ::exp2f(__x);
#endif // !_CCCL_BUILTIN_EXP2F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double exp2(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXP2)
  return _CCCL_BUILTIN_EXP2(__x);
#else // ^^^ _CCCL_BUILTIN_EXP2 ^^^ // vvv !_CCCL_BUILTIN_EXP2 vvv
  return ::exp2(__x);
#endif // !_CCCL_BUILTIN_EXP2
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double exp2(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXP2L)
  return _CCCL_BUILTIN_EXP2L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXP2L ^^^ // vvv !_CCCL_BUILTIN_EXP2L vvv
  return ::exp2l(__x);
#  endif // !_CCCL_BUILTIN_EXP2L
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double exp2l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXP2L)
  return _CCCL_BUILTIN_EXP2L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXP2L ^^^ // vvv !_CCCL_BUILTIN_EXP2L vvv
  return ::exp2l(__x);
#  endif // !_CCCL_BUILTIN_EXP2L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half exp2(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hexp2(__x);), (return __float2half(_CUDA_VSTD::exp2f(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 exp2(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hexp2(__x);), (return __float2bfloat16(_CUDA_VSTD::exp2f(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double exp2(_Integer __x) noexcept
{
  return _CUDA_VSTD::exp2((double) __x);
}

// expm1

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float expm1(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPM1F)
  return _CCCL_BUILTIN_EXPM1F(__x);
#else // ^^^ _CCCL_BUILTIN_EXPM1F ^^^ // vvv !_CCCL_BUILTIN_EXPM1F vvv
  return ::expm1f(__x);
#endif // !_CCCL_BUILTIN_EXPM1F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float expm1f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPM1F)
  return _CCCL_BUILTIN_EXPM1F(__x);
#else // ^^^ _CCCL_BUILTIN_EXPM1F ^^^ // vvv !_CCCL_BUILTIN_EXPM1F vvv
  return ::expm1f(__x);
#endif // !_CCCL_BUILTIN_EXPM1F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double expm1(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_EXPM1)
  return _CCCL_BUILTIN_EXPM1(__x);
#else // ^^^ _CCCL_BUILTIN_EXPM1 ^^^ // vvv !_CCCL_BUILTIN_EXPM1 vvv
  return ::expm1(__x);
#endif // !_CCCL_BUILTIN_EXPM1
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double expm1(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPM1L)
  return _CCCL_BUILTIN_EXPM1L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPM1L ^^^ // vvv !_CCCL_BUILTIN_EXPM1L vvv
  return ::expm1l(__x);
#  endif // !_CCCL_BUILTIN_EXPM1L
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double expm1l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_EXPM1L)
  return _CCCL_BUILTIN_EXPM1L(__x);
#  else // ^^^ _CCCL_BUILTIN_EXPM1L ^^^ // vvv !_CCCL_BUILTIN_EXPM1L vvv
  return ::expm1l(__x);
#  endif // !_CCCL_BUILTIN_EXPM1L
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half expm1(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::expm1f(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 expm1(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::expm1f(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double expm1(_Integer __x) noexcept
{
  return _CUDA_VSTD::expm1((double) __x);
}

// scalbln

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float scalbln(float __x, long __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBLNF)
  return _CCCL_BUILTIN_SCALBLNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBLNF ^^^ // vvv !_CCCL_BUILTIN_SCALBLNF vvv
  return ::scalblnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBLNF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float scalblnf(float __x, long __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBLNF)
  return _CCCL_BUILTIN_SCALBLNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBLNF ^^^ // vvv !_CCCL_BUILTIN_SCALBLNF vvv
  return ::scalblnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBLNF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double scalbln(double __x, long __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBLN)
  return _CCCL_BUILTIN_SCALBLN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBLN ^^^ // vvv !_CCCL_BUILTIN_SCALBLN vvv
  return ::scalbln(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBLN
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double scalbln(long double __x, long __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBLNL)
  return _CCCL_BUILTIN_SCALBLNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBLNL ^^^ // vvv !_CCCL_BUILTIN_SCALBLNL vvv
  return ::scalblnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBLNL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double scalblnl(long double __x, long __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBLNL)
  return _CCCL_BUILTIN_SCALBLNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBLNL ^^^ // vvv !_CCCL_BUILTIN_SCALBLNL vvv
  return ::scalblnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBLNL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half scalbln(__half __x, long __y) noexcept
{
  return __float2half(_CUDA_VSTD::scalblnf(__half2float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 scalbln(__nv_bfloat16 __x, long __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::scalblnf(__bfloat162float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double scalbln(_Integer __x, long __y) noexcept
{
  return _CUDA_VSTD::scalbln((double) __x, __y);
}

// scalbn

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float scalbn(float __x, int __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBNF)
  return _CCCL_BUILTIN_SCALBNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBNF ^^^ // vvv !_CCCL_BUILTIN_SCALBNF vvv
  return ::scalbnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBNF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float scalbnf(float __x, int __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBNF)
  return _CCCL_BUILTIN_SCALBNF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBNF ^^^ // vvv !_CCCL_BUILTIN_SCALBNF vvv
  return ::scalbnf(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBNF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double scalbn(double __x, int __y) noexcept
{
#if defined(_CCCL_BUILTIN_SCALBN)
  return _CCCL_BUILTIN_SCALBN(__x, __y);
#else // ^^^ _CCCL_BUILTIN_SCALBN ^^^ // vvv !_CCCL_BUILTIN_SCALBN vvv
  return ::scalbn(__x, __y);
#endif // !_CCCL_BUILTIN_SCALBN
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double scalbn(long double __x, int __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBNL)
  return _CCCL_BUILTIN_SCALBNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBNL ^^^ // vvv !_CCCL_BUILTIN_SCALBNL vvv
  return ::scalbnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBNL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double scalbnl(long double __x, int __y) noexcept
{
#  if defined(_CCCL_BUILTIN_SCALBNL)
  return _CCCL_BUILTIN_SCALBNL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SCALBNL ^^^ // vvv !_CCCL_BUILTIN_SCALBNL vvv
  return ::scalbnl(__x, __y);
#  endif // !_CCCL_BUILTIN_SCALBNL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half scalbn(__half __x, int __y) noexcept
{
  return __float2half(_CUDA_VSTD::scalbnf(__half2float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 scalbn(__nv_bfloat16 __x, int __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::scalbnf(__bfloat162float(__x), __y));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double scalbn(_Integer __x, int __y) noexcept
{
  return _CUDA_VSTD::scalbn((double) __x, __y);
}

// pow

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float pow(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_POWF)
  return _CCCL_BUILTIN_POWF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_POWF ^^^ // vvv !_CCCL_BUILTIN_POWF vvv
  return ::powf(__x, __y);
#endif // !_CCCL_BUILTIN_POWF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float powf(float __x, float __y) noexcept
{
#if defined(_CCCL_BUILTIN_POWF)
  return _CCCL_BUILTIN_POWF(__x, __y);
#else // ^^^ _CCCL_BUILTIN_POWF ^^^ // vvv !_CCCL_BUILTIN_POWF vvv
  return ::powf(__x, __y);
#endif // !_CCCL_BUILTIN_POWF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double pow(double __x, double __y) noexcept
{
#if defined(_CCCL_BUILTIN_POW)
  return _CCCL_BUILTIN_POW(__x, __y);
#else // ^^^ _CCCL_BUILTIN_POW ^^^ // vvv !_CCCL_BUILTIN_POW vvv
  return ::pow(__x, __y);
#endif // !_CCCL_BUILTIN_POW
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double pow(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_POWL)
  return _CCCL_BUILTIN_POWL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_POWL ^^^ // vvv !_CCCL_BUILTIN_POWL vvv
  return ::powl(__x, __y);
#  endif // !_CCCL_BUILTIN_POWL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double powl(long double __x, long double __y) noexcept
{
#  if defined(_CCCL_BUILTIN_POWL)
  return _CCCL_BUILTIN_POWL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_POWL ^^^ // vvv !_CCCL_BUILTIN_POWL vvv
  return ::powl(__x, __y);
#  endif // !_CCCL_BUILTIN_POWL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half pow(__half __x, __half __y) noexcept
{
  return __float2half(_CUDA_VSTD::powf(__half2float(__x), __half2float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 pow(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::powf(__bfloat162float(__x), __bfloat162float(__y)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _A1, class _A2, enable_if_t<_CCCL_TRAIT(is_arithmetic, _A1) && _CCCL_TRAIT(is_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __promote_t<_A1, _A2> pow(_A1 __x, _A2 __y) noexcept
{
  using __result_type = __promote_t<_A1, _A2>;
  static_assert(!(_CCCL_TRAIT(is_same, _A1, __result_type) && _CCCL_TRAIT(is_same, _A2, __result_type)), "");
  return _CUDA_VSTD::pow((__result_type) __x, (__result_type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MATH_EXPONENTIAL_FUNCTIONS_H
