//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_SATURATING_OVERFLOW_CAST_H
#define _CUDA___NUMERIC_SATURATING_OVERFLOW_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Up> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr overflow_result<_Up> saturating_overflow_cast(_Tp __x) noexcept
{
  if (::cuda::std::cmp_less(__x, ::cuda::std::numeric_limits<_Up>::min()))
  {
    return {::cuda::std::numeric_limits<_Up>::min(), true};
  }
  if (::cuda::std::cmp_greater(__x, ::cuda::std::numeric_limits<_Up>::max()))
  {
    return {::cuda::std::numeric_limits<_Up>::max(), true};
  }
  return {static_cast<_Up>(__x), false};
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_SATURATING_OVERFLOW_CAST_H
