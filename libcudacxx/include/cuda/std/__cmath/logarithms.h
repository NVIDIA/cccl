// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_LOGARITHMS_H
#define _LIBCUDACXX___CMATH_LOGARITHMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/common.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/cstdint>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// log

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGF)
  return _CCCL_BUILTIN_LOGF(__x);
#else // ^^^ _CCCL_BUILTIN_LOGF ^^^ / vvv !_CCCL_BUILTIN_LOGF vvv
  return ::logf(__x);
#endif // !_CCCL_BUILTIN_LOGF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float logf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGF)
  return _CCCL_BUILTIN_LOGF(__x);
#else // ^^^ _CCCL_BUILTIN_LOGF ^^^ / vvv !_CCCL_BUILTIN_LOGF vvv
  return ::logf(__x);
#endif // !_CCCL_BUILTIN_LOGF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG)
  return _CCCL_BUILTIN_LOG(__x);
#else // ^^^ _CCCL_BUILTIN_LOG ^^^ / vvv !_CCCL_BUILTIN_LOG vvv
  return ::log(__x);
#endif // !_CCCL_BUILTIN_LOG
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOGL)
  return _CCCL_BUILTIN_LOGL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOGL ^^^ / vvv !_CCCL_BUILTIN_LOGL vvv
  return ::logl(__x);
#  endif // !_CCCL_BUILTIN_LOGL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double logl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOGL)
  return _CCCL_BUILTIN_LOGL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOGL ^^^ / vvv !_CCCL_BUILTIN_LOGL vvv
  return ::logl(__x);
#  endif // !_CCCL_BUILTIN_LOGL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half log(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (return ::hlog(__x);), ({
                      float __vf            = __half2float(__x);
                      __vf                  = _CUDA_VSTD::logf(__vf);
                      __half_raw __ret_repr = ::__float2half_rn(__vf);

                      _CUDA_VSTD::uint16_t __repr = __half_raw(__x).x;
                      switch (__repr)
                      {
                        case 7544:
                          __ret_repr.x -= 1;
                          break;

                        default:;
                      }

                      return __ret_repr;
                    }))
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 log(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hlog(__x);), (return __float2bfloat16(_CUDA_VSTD::logf(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG)
  return _CCCL_BUILTIN_LOG((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG ^^^ / vvv !_CCCL_BUILTIN_LOG vvv
  return ::log((double) __x);
#endif // !_CCCL_BUILTIN_LOG
}

// log10

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log10(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10F)
  return _CCCL_BUILTIN_LOG10F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG10F ^^^ / vvv !_CCCL_BUILTIN_LOG10F vvv
  return ::log10f(__x);
#endif // !_CCCL_BUILTIN_LOG10F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log10f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10F)
  return _CCCL_BUILTIN_LOG10F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG10F ^^^ / vvv !_CCCL_BUILTIN_LOG10F vvv
  return ::log10f(__x);
#endif // !_CCCL_BUILTIN_LOG10F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log10(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10)
  return _CCCL_BUILTIN_LOG10(__x);
#else // ^^^ _CCCL_BUILTIN_LOG10 ^^^ / vvv !_CCCL_BUILTIN_LOG10 vvv
  return ::log10(__x);
#endif // !_CCCL_BUILTIN_LOG10
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log10(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG10L)
  return _CCCL_BUILTIN_LOG10L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG10L ^^^ / vvv !_CCCL_BUILTIN_LOG10L vvv
  return ::log10l(__x);
#  endif // !_CCCL_BUILTIN_LOG10L
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log10l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG10L)
  return _CCCL_BUILTIN_LOG10L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG10L ^^^ / vvv !_CCCL_BUILTIN_LOG10L vvv
  return ::log10l(__x);
#  endif // !_CCCL_BUILTIN_LOG10L
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half log10(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_53, (return ::hlog10(__x);), (return __float2half(_CUDA_VSTD::log10f(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 log10(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hlog10(__x);), (return __float2bfloat16(_CUDA_VSTD::log10f(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log10(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG10)
  return _CCCL_BUILTIN_LOG10((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG10 ^^^ / vvv !_CCCL_BUILTIN_LOG10 vvv
  return ::log10((double) __x);
#endif // !_CCCL_BUILTIN_LOG10
}

// ilogb

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogb(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ILOGBF)
  return _CCCL_BUILTIN_ILOGBF(__x);
#else // ^^^ _CCCL_BUILTIN_ILOGBF ^^^ / vvv !_CCCL_BUILTIN_ILOGBF vvv
  return ::ilogbf(__x);
#endif // !_CCCL_BUILTIN_ILOGBF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogbf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ILOGBF)
  return _CCCL_BUILTIN_ILOGBF(__x);
#else // ^^^ _CCCL_BUILTIN_ILOGBF ^^^ / vvv !_CCCL_BUILTIN_ILOGBF vvv
  return ::ilogbf(__x);
#endif // !_CCCL_BUILTIN_ILOGBF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogb(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ILOGB)
  return _CCCL_BUILTIN_ILOGB(__x);
#else // ^^^ _CCCL_BUILTIN_ILOGB ^^^ / vvv !_CCCL_BUILTIN_ILOGB vvv
  return ::ilogb(__x);
#endif // !_CCCL_BUILTIN_ILOGB
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogb(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ILOGBL)
  return _CCCL_BUILTIN_ILOGBL(__x);
#  else // ^^^ _CCCL_BUILTIN_ILOGBL ^^^ / vvv !_CCCL_BUILTIN_ILOGBL vvv
  return ::ilogbl(__x);
#  endif // !_CCCL_BUILTIN_ILOGBL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogbl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ILOGBL)
  return _CCCL_BUILTIN_ILOGBL(__x);
#  else // ^^^ _CCCL_BUILTIN_ILOGBL ^^^ / vvv !_CCCL_BUILTIN_ILOGBL vvv
  return ::ilogbl(__x);
#  endif // !_CCCL_BUILTIN_ILOGBL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogb(__half __x) noexcept
{
  return _CUDA_VSTD::ilogbf(__half2float(__x));
}
#endif // defined(_LIBCUDACXX_HAS_NVFP16)

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogb(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::ilogbf(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI int ilogb(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_ILOGB)
  return _CCCL_BUILTIN_ILOGB((double) __x);
#else // ^^^ _CCCL_BUILTIN_ILOGB ^^^ / vvv !_CCCL_BUILTIN_ILOGB vvv
  return ::ilogb((double) __x);
#endif // !_CCCL_BUILTIN_ILOGB
}

// log1p

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log1p(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1PF)
  return _CCCL_BUILTIN_LOG1PF(__x);
#else // ^^^ _CCCL_BUILTIN_LOG1PF ^^^ / vvv !_CCCL_BUILTIN_LOG1PF vvv
  return ::log1pf(__x);
#endif // !_CCCL_BUILTIN_LOG1PF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log1pf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1PF)
  return _CCCL_BUILTIN_LOG1PF(__x);
#else // ^^^ _CCCL_BUILTIN_LOG1PF ^^^ / vvv !_CCCL_BUILTIN_LOG1PF vvv
  return ::log1pf(__x);
#endif // !_CCCL_BUILTIN_LOG1PF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log1p(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1P)
  return _CCCL_BUILTIN_LOG1P(__x);
#else // ^^^ _CCCL_BUILTIN_LOG1P ^^^ / vvv !_CCCL_BUILTIN_LOG1P vvv
  return ::log1p(__x);
#endif // !_CCCL_BUILTIN_LOG1P
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log1p(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG1PL)
  return _CCCL_BUILTIN_LOG1PL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG1PL ^^^ / vvv !_CCCL_BUILTIN_LOG1PL vvv
  return ::log1pl(__x);
#  endif // !_CCCL_BUILTIN_LOG1PL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log1pl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG1PL)
  return _CCCL_BUILTIN_LOG1PL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG1PL ^^^ / vvv !_CCCL_BUILTIN_LOG1PL vvv
  return ::log1pl(__x);
#  endif // !_CCCL_BUILTIN_LOG1PL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half log1p(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::log1pf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 log1p(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::log1pf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log1p(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG1P)
  return _CCCL_BUILTIN_LOG1P((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG1P ^^^ / vvv !_CCCL_BUILTIN_LOG1P vvv
  return ::log1p((double) __x);
#endif // !_CCCL_BUILTIN_LOG1P
}

// log2

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log2(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2F)
  return _CCCL_BUILTIN_LOG2F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG2F ^^^ / vvv !_CCCL_BUILTIN_LOG2F vvv
  return ::log2f(__x);
#endif // !_CCCL_BUILTIN_LOG2F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float log2f(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2F)
  return _CCCL_BUILTIN_LOG2F(__x);
#else // ^^^ _CCCL_BUILTIN_LOG2F ^^^ / vvv !_CCCL_BUILTIN_LOG2F vvv
  return ::log2f(__x);
#endif // !_CCCL_BUILTIN_LOG2F
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log2(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2)
  return _CCCL_BUILTIN_LOG2(__x);
#else // ^^^ _CCCL_BUILTIN_LOG2 ^^^ / vvv !_CCCL_BUILTIN_LOG2 vvv
  return ::log2(__x);
#endif // !_CCCL_BUILTIN_LOG2
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log2(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG2L)
  return _CCCL_BUILTIN_LOG2L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG2L ^^^ / vvv !_CCCL_BUILTIN_LOG2L vvv
  return ::log2l(__x);
#  endif // !_CCCL_BUILTIN_LOG2L
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double log2l(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOG2L)
  return _CCCL_BUILTIN_LOG2L(__x);
#  else // ^^^ _CCCL_BUILTIN_LOG2L ^^^ / vvv !_CCCL_BUILTIN_LOG2L vvv
  return ::log2l(__x);
#  endif // !_CCCL_BUILTIN_LOG2L
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half log2(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_53, (return ::hlog2(__x);), (return __float2half(_CUDA_VSTD::log2f(__half2float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 log2(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hlog2(__x);), (return __float2bfloat16(_CUDA_VSTD::log2f(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double log2(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOG2)
  return _CCCL_BUILTIN_LOG2((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOG2 ^^^ / vvv !_CCCL_BUILTIN_LOG2 vvv
  return ::log2((double) __x);
#endif // !_CCCL_BUILTIN_LOG2
}

// logb

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float logb(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGBF)
  return _CCCL_BUILTIN_LOGBF(__x);
#else // ^^^ _CCCL_BUILTIN_LOGBF ^^^ / vvv !_CCCL_BUILTIN_LOGBF vvv
  return ::logbf(__x);
#endif // !_CCCL_BUILTIN_LOGBF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float logbf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGBF)
  return _CCCL_BUILTIN_LOGBF(__x);
#else // ^^^ _CCCL_BUILTIN_LOGBF ^^^ / vvv !_CCCL_BUILTIN_LOGBF vvv
  return ::logbf(__x);
#endif // !_CCCL_BUILTIN_LOGBF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double logb(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGB)
  return _CCCL_BUILTIN_LOGB(__x);
#else // ^^^ _CCCL_BUILTIN_LOGB ^^^ / vvv !_CCCL_BUILTIN_LOGB vvv
  return ::logb(__x);
#endif // !_CCCL_BUILTIN_LOGB
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double logb(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOGBL)
  return _CCCL_BUILTIN_LOGBL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOGBL ^^^ / vvv !_CCCL_BUILTIN_LOGBL vvv
  return ::logbl(__x);
#  endif // !_CCCL_BUILTIN_LOGBL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double logbl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LOGBL)
  return _CCCL_BUILTIN_LOGBL(__x);
#  else // ^^^ _CCCL_BUILTIN_LOGBL ^^^ / vvv !_CCCL_BUILTIN_LOGBL vvv
  return ::logbl(__x);
#  endif // !_CCCL_BUILTIN_LOGBL
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half logb(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::logbf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 logb(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::logbf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double logb(_Integer __x) noexcept
{
#if defined(_CCCL_BUILTIN_LOGB)
  return _CCCL_BUILTIN_LOGB((double) __x);
#else // ^^^ _CCCL_BUILTIN_LOGB ^^^ / vvv !_CCCL_BUILTIN_LOGB vvv
  return ::logb((double) __x);
#endif // !_CCCL_BUILTIN_LOGB
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_LOGARITHMS_H
