// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_GAMMA_H
#define _LIBCUDACXX___CMATH_GAMMA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/nvfp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <nv/target>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// lgamma

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float lgamma(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LGAMMAF)
  return _CCCL_BUILTIN_LGAMMAF(__x);
#else // ^^^ _CCCL_BUILTIN_LGAMMAF ^^^ / vvv !_CCCL_BUILTIN_LGAMMAF vvv
  return ::lgammaf(__x);
#endif // !_CCCL_BUILTIN_LGAMMAF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float lgammaf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_LGAMMAF)
  return _CCCL_BUILTIN_LGAMMAF(__x);
#else // ^^^ _CCCL_BUILTIN_LGAMMAF ^^^ / vvv !_CCCL_BUILTIN_LGAMMAF vvv
  return ::lgammaf(__x);
#endif // !_CCCL_BUILTIN_LGAMMAF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double lgamma(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_LGAMMA)
  return _CCCL_BUILTIN_LGAMMA(__x);
#else // ^^^ _CCCL_BUILTIN_LGAMMA ^^^ / vvv !_CCCL_BUILTIN_LGAMMA vvv
  return ::lgamma(__x);
#endif // !_CCCL_BUILTIN_LGAMMA
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double lgamma(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LGAMMAL)
  return _CCCL_BUILTIN_LGAMMAL(__x);
#  else // ^^^ _CCCL_BUILTIN_LGAMMAL ^^^ / vvv !_CCCL_BUILTIN_LGAMMAL vvv
  return ::lgammal(__x);
#  endif // !_CCCL_BUILTIN_LGAMMAL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double lgammal(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_LGAMMAL)
  return _CCCL_BUILTIN_LGAMMAL(__x);
#  else // ^^^ _CCCL_BUILTIN_LGAMMAL ^^^ / vvv !_CCCL_BUILTIN_LGAMMAL vvv
  return ::lgammal(__x);
#  endif // !_CCCL_BUILTIN_LGAMMAL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half lgamma(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::lgammaf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 lgamma(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::lgammaf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double lgamma(_Integer __x) noexcept
{
  return _CUDA_VSTD::lgamma((double) __x);
}

// tgamma

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float tgamma(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TGAMMAF)
  return _CCCL_BUILTIN_TGAMMAF(__x);
#else // ^^^ _CCCL_BUILTIN_TGAMMAF ^^^ / vvv !_CCCL_BUILTIN_TGAMMAF vvv
  return ::tgammaf(__x);
#endif // !_CCCL_BUILTIN_TGAMMAF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float tgammaf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TGAMMAF)
  return _CCCL_BUILTIN_TGAMMAF(__x);
#else // ^^^ _CCCL_BUILTIN_TGAMMAF ^^^ / vvv !_CCCL_BUILTIN_TGAMMAF vvv
  return ::tgammaf(__x);
#endif // !_CCCL_BUILTIN_TGAMMAF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double tgamma(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_TGAMMA)
  return _CCCL_BUILTIN_TGAMMA(__x);
#else // ^^^ _CCCL_BUILTIN_TGAMMA ^^^ / vvv !_CCCL_BUILTIN_TGAMMA vvv
  return ::tgamma(__x);
#endif // !_CCCL_BUILTIN_TGAMMA
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double tgamma(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TGAMMAL)
  return _CCCL_BUILTIN_TGAMMAL(__x);
#  else // ^^^ _CCCL_BUILTIN_TGAMMAL ^^^ / vvv !_CCCL_BUILTIN_TGAMMAL vvv
  return ::tgammal(__x);
#  endif // !_CCCL_BUILTIN_TGAMMAL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double tgammal(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TGAMMAL)
  return _CCCL_BUILTIN_TGAMMAL(__x);
#  else // ^^^ _CCCL_BUILTIN_TGAMMAL ^^^ / vvv !_CCCL_BUILTIN_TGAMMAL vvv
  return ::tgammal(__x);
#  endif // !_CCCL_BUILTIN_TGAMMAL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half tgamma(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::tgammaf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 tgamma(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::tgammaf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double tgamma(_Integer __x) noexcept
{
  return _CUDA_VSTD::tgamma((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_GAMMA_H
