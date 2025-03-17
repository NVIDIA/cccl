// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_TRIGONOMETRIC_FUNCTIONS_H
#define _LIBCUDACXX___CMATH_TRIGONOMETRIC_FUNCTIONS_H

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
#include <cuda/std/cstdint>

#include <nv/target>

// MSVC and clang cuda need the host side functions included
#if _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)
#  include <math.h>
#endif // _CCCL_COMPILER(MSVC) || _CCCL_CUDA_COMPILER(CLANG)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// cos

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float cos(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_COSF)
  return _CCCL_BUILTIN_COSF(__x);
#else // ^^^ _CCCL_BUILTIN_COSF ^^^ / vvv !_CCCL_BUILTIN_COSF vvv
  return ::cosf(__x);
#endif // !_CCCL_BUILTIN_COSF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float cosf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_COSF)
  return _CCCL_BUILTIN_COSF(__x);
#else // ^^^ _CCCL_BUILTIN_COSF ^^^ / vvv !_CCCL_BUILTIN_COSF vvv
  return ::cosf(__x);
#endif // !_CCCL_BUILTIN_COSF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double cos(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_COS)
  return _CCCL_BUILTIN_COS(__x);
#else // ^^^ _CCCL_BUILTIN_COS ^^^ / vvv !_CCCL_BUILTIN_COS vvv
  return ::cos(__x);
#endif // !_CCCL_BUILTIN_COS
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double cos(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_COSL)
  return _CCCL_BUILTIN_COSL(__x);
#  else // ^^^ _CCCL_BUILTIN_COSL ^^^ / vvv !_CCCL_BUILTIN_COSL vvv
  return ::cosl(__x);
#  endif // !_CCCL_BUILTIN_COSL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double cosl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_COSL)
  return _CCCL_BUILTIN_COSL(__x);
#  else // ^^^ _CCCL_BUILTIN_COSL ^^^ / vvv !_CCCL_BUILTIN_COSL vvv
  return ::cosl(__x);
#  endif // !_CCCL_BUILTIN_COSL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half cos(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (return ::hcos(__x);), ({
                      float __xf            = __half2float(__x);
                      __xf                  = _CUDA_VSTD::cosf(__xf);
                      __half_raw __ret_repr = ::__float2half_rn(__xf);

                      uint16_t __repr = __half_raw(__x).x;
                      switch (__repr)
                      {
                        case 11132:
                        case 43900:
                          __ret_repr.x += 1;
                          break;

                        default:;
                      }

                      return __ret_repr;
                    }))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 cos(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hcos(__x);), (return __float2bfloat16(_CUDA_VSTD::cosf(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double cos(_Integer __x) noexcept
{
  return _CUDA_VSTD::cos((double) __x);
}

// sin

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float sin(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SINF)
  return _CCCL_BUILTIN_SINF(__x);
#else // ^^^ _CCCL_BUILTIN_SINF ^^^ / vvv !_CCCL_BUILTIN_SINF vvv
  return ::sinf(__x);
#endif // !_CCCL_BUILTIN_SINF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float sinf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SINF)
  return _CCCL_BUILTIN_SINF(__x);
#else // ^^^ _CCCL_BUILTIN_SINF ^^^ / vvv !_CCCL_BUILTIN_SINF vvv
  return ::sinf(__x);
#endif // !_CCCL_BUILTIN_SINF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double sin(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIN)
  return _CCCL_BUILTIN_SIN(__x);
#else // ^^^ _CCCL_BUILTIN_SIN ^^^ / vvv !_CCCL_BUILTIN_SIN vvv
  return ::sin(__x);
#endif // !_CCCL_BUILTIN_SIN
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double sin(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SINL)
  return _CCCL_BUILTIN_SINL(__x);
#  else // ^^^ _CCCL_BUILTIN_SINL ^^^ / vvv !_CCCL_BUILTIN_SINL vvv
  return ::sinl(__x);
#  endif // !_CCCL_BUILTIN_SINL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double sinl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SINL)
  return _CCCL_BUILTIN_SINL(__x);
#  else // ^^^ _CCCL_BUILTIN_SINL ^^^ / vvv !_CCCL_BUILTIN_SINL vvv
  return ::sinl(__x);
#  endif // !_CCCL_BUILTIN_SINL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half sin(__half __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (return ::hsin(__x);), ({
                      float __xf            = __half2float(__x);
                      __xf                  = _CUDA_VSTD::sinf(__xf);
                      __half_raw __ret_repr = ::__float2half_rn(__xf);

                      uint16_t __repr = __half_raw(__x).x;
                      switch (__repr)
                      {
                        case 12979:
                        case 45747:
                          __ret_repr.x += 1;
                          break;

                        case 23728:
                        case 56496:
                          __ret_repr.x -= 1;
                          break;

                        default:;
                      }

                      return __ret_repr;
                    }))
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 sin(__nv_bfloat16 __x) noexcept
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (return ::hsin(__x);), (return __float2bfloat16(_CUDA_VSTD::sinf(__bfloat162float(__x)));))
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double sin(_Integer __x) noexcept
{
  return _CUDA_VSTD::sin((double) __x);
}

// tan

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float tan(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TANF)
  return _CCCL_BUILTIN_TANF(__x);
#else // ^^^ _CCCL_BUILTIN_TANF ^^^ / vvv !_CCCL_BUILTIN_TANF vvv
  return ::tanf(__x);
#endif // !_CCCL_BUILTIN_TANF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI float tanf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_TANF)
  return _CCCL_BUILTIN_TANF(__x);
#else // ^^^ _CCCL_BUILTIN_TANF ^^^ / vvv !_CCCL_BUILTIN_TANF vvv
  return ::tanf(__x);
#endif // !_CCCL_BUILTIN_TANF
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double tan(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_TAN)
  return _CCCL_BUILTIN_TAN(__x);
#else // ^^^ _CCCL_BUILTIN_TAN ^^^ / vvv !_CCCL_BUILTIN_TAN vvv
  return ::tan(__x);
#endif // !_CCCL_BUILTIN_TAN
}

#if _CCCL_HAS_LONG_DOUBLE()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double tan(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TANL)
  return _CCCL_BUILTIN_TANL(__x);
#  else // ^^^ _CCCL_BUILTIN_TANL ^^^ / vvv !_CCCL_BUILTIN_TANL vvv
  return ::tanl(__x);
#  endif // !_CCCL_BUILTIN_TANL
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI long double tanl(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_TANL)
  return _CCCL_BUILTIN_TANL(__x);
#  else // ^^^ _CCCL_BUILTIN_TANL ^^^ / vvv !_CCCL_BUILTIN_TANL vvv
  return ::tanl(__x);
#  endif // !_CCCL_BUILTIN_TANL
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __half tan(__half __x) noexcept
{
  return __float2half(_CUDA_VSTD::tanf(__half2float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 tan(__nv_bfloat16 __x) noexcept
{
  return __float2bfloat16(_CUDA_VSTD::tanf(__bfloat162float(__x)));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

template <class _Integer, enable_if_t<_CCCL_TRAIT(is_integral, _Integer), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI double tan(_Integer __x) noexcept
{
  return _CUDA_VSTD::tan((double) __x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_TRIGONOMETRIC_FUNCTIONS_H
