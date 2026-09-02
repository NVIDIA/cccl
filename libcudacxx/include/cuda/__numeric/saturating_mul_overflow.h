//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_SATURATING_MUL_OVERFLOW_H
#define _CUDA___NUMERIC_SATURATING_MUL_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/mul_overflow.h>
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> saturating_mul_overflow(_Tp __x, _Tp __y) noexcept
{
  auto __result = ::cuda::mul_overflow(__x, __y);
  if (__result.overflow)
  {
    if constexpr (::cuda::std::is_signed_v<_Tp>)
    {
      __result.value =
        ((__x < 0) == (__y < 0)) ? ::cuda::std::numeric_limits<_Tp>::max() : ::cuda::std::numeric_limits<_Tp>::min();
    }
    else
    {
      __result.value = ::cuda::std::numeric_limits<_Tp>::max();
    }
  }
  return __result;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool saturating_mul_overflow(_Tp& __result, _Tp __x, _Tp __y) noexcept
{
  const auto [__value, __overflow] = ::cuda::saturating_mul_overflow(__x, __y);
  __result                         = __value;
  return __overflow;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_SATURATING_MUL_OVERFLOW_H
