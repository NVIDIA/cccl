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

#  include <cuda/std/__internal/nvfp_types.h>
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

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 hypot(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __float2bfloat16(::hypotf(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 atan2(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __float2bfloat16(::atan2f(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __nv_bfloat16 __constexpr_fmax(__nv_bfloat16 __x, __nv_bfloat16 __y) noexcept
{
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVBF16

#endif // _LIBCUDACXX___CMATH_NVBF16_H
