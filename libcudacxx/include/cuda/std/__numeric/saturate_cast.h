// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_SATURATE_CAST_H
#define _LIBCUDACXX___NUMERIC_SATURATE_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Up> _CCCL_AND __cccl_is_integer_v<_Tp>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Up saturate_cast(_Tp __x) noexcept
{
  if (_CUDA_VSTD::cmp_less(__x, numeric_limits<_Up>::min()))
  {
    return numeric_limits<_Up>::min();
  }
  if (_CUDA_VSTD::cmp_greater(__x, numeric_limits<_Up>::max()))
  {
    return numeric_limits<_Up>::max();
  }
  return static_cast<_Up>(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NUMERIC_SATURATE_CAST_H
