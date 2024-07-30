// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_CMATH_NVBF16_H
#define _LIBCUDACXX___CUDA_CMATH_NVBF16_H

#include <cuda/std/detail/__config>

_CCCL_IMPLICIT_SYSTEM_HEADER

#if defined(_LIBCUDACXX_HAS_NVBF16)

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP

#  include <cuda/std/cstdint>

#  include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// trigonometric functions
inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 sin(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsin(__v);), (return __float2bfloat16(::sin(__bfloat162float(__v)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 sinh(__nv_bfloat16 __v)
{
  return __float2bfloat16(::sinh(__bfloat162float(__v)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 cos(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hcos(__v);), (return __float2bfloat16(::cos(__bfloat162float(__v)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 cosh(__nv_bfloat16 __v)
{
  return __float2bfloat16(::cosh(__bfloat162float(__v)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 exp(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hexp(__v);), (return __float2bfloat16(::exp(__bfloat162float(__v)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 hypot(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __float2bfloat16(::hypot(__bfloat162float(__x), __bfloat162float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 atan2(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __float2bfloat16(::atan2(__bfloat162float(__x), __bfloat162float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 log(__nv_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hlog(__x);), (return __float2bfloat16(::log(__bfloat162float(__x)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 sqrt(__nv_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsqrt(__x);), (return __float2bfloat16(::sqrt(__bfloat162float(__x)));))
}

// floating point helper
inline _LIBCUDACXX_INLINE_VISIBILITY bool signbit(__nv_bfloat16 __v)
{
  return ::signbit(::__bfloat162float(__v));
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool __constexpr_isnan(__nv_bfloat16 __x) noexcept
{
  return ::__hisnan(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool isnan(__nv_bfloat16 __v)
{
  return __constexpr_isnan(__v);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool __constexpr_isinf(__nv_bfloat16 __x) noexcept
{
#  if _CCCL_STD_VER >= 2020 && defined(_CCCL_CUDACC_BELOW_12_3)
  // this is a workaround for nvbug 4362808
  return !::__hisnan(__x) && ::__hisnan(__x - __x);
#  else // ^^^ C++20 && below 12.3 ^^^ / vvv C++17 or 12.3+ vvv
  return ::__hisinf(__x) != 0;
#  endif // _CCCL_STD_VER <= 2017 || _CCCL_CUDACC_VER < 1203000
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool isinf(__nv_bfloat16 __v)
{
  return __constexpr_isinf(__v);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool __constexpr_isfinite(__nv_bfloat16 __x) noexcept
{
  return !__constexpr_isnan(__x) && !__constexpr_isinf(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY bool isfinite(__nv_bfloat16 __v)
{
  return __constexpr_isfinite(__v);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 __constexpr_copysign(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(::copysignf(__bfloat162float(__x), __bfloat162float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 copysign(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __constexpr_copysign(__x, __y);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 __constexpr_fabs(__nv_bfloat16 __x) noexcept
{
  return ::__habs(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 fabs(__nv_bfloat16 __x)
{
  return __constexpr_fabs(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 abs(__nv_bfloat16 __x)
{
  return __constexpr_fabs(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 __constexpr_fmax(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVBF16

#endif // _LIBCUDACXX___CUDA_CMATH_NVBF16_H
