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

#if defined(_LIBCUDACXX_HAS_NVBF16)

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP

#  include <nv/target>

#  include "../__type_traits/integral_constant.h"
#  include "../cmath"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// trigonometric functions
inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 sin(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsin(__v);), (return __nv_bfloat16(_CUDA_VSTD::sin(float(__v)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 sinh(__nv_bfloat16 __v)
{
  return __nv_bfloat16(_CUDA_VSTD::sinh(float(__v)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 cos(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return hcos(__v);), (return __nv_bfloat16(_CUDA_VSTD::cos(float(__v)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 cosh(__nv_bfloat16 __v)
{
  return __nv_bfloat16(_CUDA_VSTD::cosh(float(__v)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 exp(__nv_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hexp(__v);), (return __nv_bfloat16(_CUDA_VSTD::exp(float(__v)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 hypot(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __nv_bfloat16(_CUDA_VSTD::hypot(float(__x), float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 atan2(__nv_bfloat16 __x, __nv_bfloat16 __y)
{
  return __nv_bfloat16(_CUDA_VSTD::atan2(float(__x), float(__y)));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 log(__nv_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hlog(__x);), (return __nv_bfloat16(_CUDA_VSTD::log(float(__x)));))
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 sqrt(__nv_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::hsqrt(__x);), (return __nv_bfloat16(_CUDA_VSTD::sqrt(float(__x)));))
}

// floating point helper
inline _LIBCUDACXX_INLINE_VISIBILITY bool signbit(__nv_bfloat16 __v)
{
  return ::signbit(::__bfloat162float(__v));
}

inline _LIBCUDACXX_INLINE_VISIBILITY __nv_bfloat16 abs(__nv_bfloat16 __x)
{
  return __constexpr_fabs(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVBF16

#endif // _LIBCUDACXX___CUDA_CMATH_NVBF16_H
