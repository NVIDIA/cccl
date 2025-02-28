// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_NVFP16_H
#define _LIBCUDACXX___CMATH_NVFP16_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_NVFP16)

#  include <cuda_fp16.h>

#  include <cuda/std/cstdint>

#  include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// trigonometric functions

_LIBCUDACXX_HIDE_FROM_ABI __half hypot(__half __x, __half __y)
{
  return __float2half(::hypotf(__half2float(__x), __half2float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __half atan2(__half __x, __half __y)
{
  return __float2half(::atan2f(__half2float(__x), __half2float(__y)));
}

// floating point helper
_LIBCUDACXX_HIDE_FROM_ABI __half __constexpr_copysign(__half __x, __half __y) noexcept
{
  return __float2half(::copysignf(__half2float(__x), __half2float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __half copysign(__half __x, __half __y)
{
  return _CUDA_VSTD::__constexpr_copysign(__x, __y);
}

_LIBCUDACXX_HIDE_FROM_ABI __half __constexpr_fabs(__half __x) noexcept
{
  return ::__habs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __half fabs(__half __x)
{
  return _CUDA_VSTD::__constexpr_fabs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __half abs(__half __x)
{
  return _CUDA_VSTD::__constexpr_fabs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __half __constexpr_fmax(__half __x, __half __y) noexcept
{
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVFP16

#endif // _LIBCUDACXX___CMATH_NVFP16_H
