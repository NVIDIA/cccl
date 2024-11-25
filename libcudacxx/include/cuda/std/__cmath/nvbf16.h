// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_NVBF16_H
#define _LIBCUDACXX___CMATH_NVBF16_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_NVBF16)

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP

#  include <cuda/std/cstdint>

#  include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// trigonometric functions
_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 sin(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsin(__v);), (return __float2bfloat16(::sinf(__bfloat162float(__v)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 sinh(__nv_bfloat16 __v)
{
  return __float2bfloat16(::sinhf(__bfloat162float(__v)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 cos(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hcos(__v);), (return __float2bfloat16(::cosf(__bfloat162float(__v)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 cosh(__nv_bfloat16 __v)
{
  return __float2bfloat16(::coshf(__bfloat162float(__v)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 exp(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hexp(__v);), (return __float2bfloat16(::expf(__bfloat162float(__v)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 hypot(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __float2bfloat16(::hypotf(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 atan2(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __float2bfloat16(::atan2f(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 sqrt(__nv_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsqrt(__x);), (return __float2bfloat16(::sqrtf(__bfloat162float(__x)));))
}

// floating point helper
_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __constexpr_copysign(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return __float2bfloat16(::copysignf(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 copysign(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return _CUDA_VSTD::__constexpr_copysign(__x, __y);
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __constexpr_fabs(__nv_bfloat16 __x) noexcept
{
  return ::__habs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 fabs(__nv_bfloat16 __x)
{
  return _CUDA_VSTD::__constexpr_fabs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 abs(__nv_bfloat16 __x)
{
  return _CUDA_VSTD::__constexpr_fabs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __constexpr_fmax(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVBF16

#endif // _LIBCUDACXX___CMATH_NVBF16_H
