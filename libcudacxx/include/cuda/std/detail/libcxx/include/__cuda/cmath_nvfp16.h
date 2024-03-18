// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_CMATH_NVFP16_H
#define _LIBCUDACXX___CUDA_CMATH_NVFP16_H

#ifndef __cuda_std__
#  include <config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_NVFP16)

#  include <cuda_fp16.h>

#  include <nv/target>

#  include "../cmath"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// trigonometric functions
inline _LIBCUDACXX_INLINE_VISIBILITY __half sin(__half __v)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (
    return ::hsin(__v);
  ), (
    {
      float __vf            = __v;
      __vf                  = ::sin(__vf);
      __half_raw __ret_repr = ::__float2half_rn(__vf);

      uint16_t __repr = __half_raw(__v).x;
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
  }
))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half sinh(__half __v)
{
  return __half(::sinh(float(__v)));
}

// clang-format off
inline _LIBCUDACXX_INLINE_VISIBILITY __half cos(__half __v)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (
    return ::hcos(__v);
  ), (
    {
      float __vf            = __v;
      __vf                  = ::cos(__vf);
      __half_raw __ret_repr = ::__float2half_rn(__vf);

      uint16_t __repr = __half_raw(__v).x;
      switch (__repr)
      {
        case 11132:
        case 43900:
          __ret_repr.x += 1;
          break;

        default:;
      }

      return __ret_repr;
    }
  ))
}
// clang-format on

inline _LIBCUDACXX_INLINE_VISIBILITY __half cosh(__half __v)
{
  return __half(::cosh(float(__v)));
}

// clang-format off
inline _LIBCUDACXX_INLINE_VISIBILITY __half exp(__half __v)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (
    return ::hexp(__v);
  ), (
    {
      float __vf            = __v;
      __vf                  = ::exp(__vf);
      __half_raw __ret_repr = ::__float2half_rn(__vf);

      uint16_t __repr = __half_raw(__v).x;
      switch (__repr)
      {
        case 8057:
        case 9679:
          __ret_repr.x -= 1;
          break;

        default:;
      }

      return __ret_repr;
    }
  ))
}
// clang-format on

inline _LIBCUDACXX_INLINE_VISIBILITY __half hypot(__half __x, __half __y)
{
  return __half(::hypot(float(__x), float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half atan2(__half __x, __half __y)
{
  return __half(::atan2(float(__x), float(__y)));
}

// clang-format off
inline _LIBCUDACXX_INLINE_VISIBILITY __half log(__half __x)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_53, (
    return ::hlog(__x);
  ), (
    {
      float __vf            = __x;
      __vf                  = ::log(__vf);
      __half_raw __ret_repr = ::__float2half_rn(__vf);

      uint16_t __repr = __half_raw(__x).x;
      switch (__repr)
      {
        case 7544:
          __ret_repr.x -= 1;
          break;

        default:;
      }

      return __ret_repr;
    }
  ))
}
// clang-format on

inline _LIBCUDACXX_INLINE_VISIBILITY __half sqrt(__half __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsqrt(__x);), (return __half(::sqrt(float(__x)));))
}

// floating point helper
inline _LIBCUDACXX_INLINE_VISIBILITY bool signbit(__half __v)
{
  return ::signbit(::__half2float(__v));
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool __constexpr_isnan(__half __x) noexcept
{
  return ::__hisnan(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool isnan(__half __v)
{
  return __constexpr_isnan(__v);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool __constexpr_isinf(__half __x) noexcept
{
#  if _CCCL_STD_VER >= 2020 && defined(_LIBCUDACXX_CUDACC_BELOW_12_3)
  // this is a workaround for nvbug 4362808
  return !::__hisnan(__x) && ::__hisnan(__x - __x);
#  else // ^^^ C++20 && below 12.3 ^^^ / vvv C++17 or 12.3+ vvv
  return ::__hisinf(__x) != 0;
#  endif // _CCCL_STD_VER <= 2017 || _LIBCUDACXX_CUDACC_VER < 1203000
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool isinf(__half __v)
{
  return __constexpr_isinf(__v);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool __constexpr_isfinite(__half __x) noexcept
{
  return !__constexpr_isnan(__x) && !__constexpr_isinf(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool isfinite(__half __v)
{
  return __constexpr_isfinite(__v);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half __constexpr_copysign(__half __x, __half __y) noexcept
{
  return __half(::copysignf(float(__x), float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half copysign(__half __x, __half __y)
{
  return __constexpr_copysign(__x, __y);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half __constexpr_fabs(__half __x) noexcept
{
  return ::__habs(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half fabs(__half __x)
{
  return __constexpr_fabs(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half abs(__half __x)
{
  return __constexpr_fabs(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __half __constexpr_fmax(__half __x, __half __y) noexcept
{
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVFP16

#endif // _LIBCUDACXX___CUDA_CMATH_NVFP16_H
