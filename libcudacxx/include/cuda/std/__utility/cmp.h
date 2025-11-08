//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_CMP_H
#define _CUDA_STD___UTILITY_CMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool cmp_equal(_Tp __t, _Up __u) noexcept
{
  if constexpr (is_signed_v<_Tp> == is_signed_v<_Up>)
  {
    return __t == __u;
  }
  else if constexpr (is_signed_v<_Tp>)
  {
    return __t < 0 ? false : make_unsigned_t<_Tp>(__t) == __u;
  }
  else
  {
    return __u < 0 ? false : __t == make_unsigned_t<_Up>(__u);
  }
  _CCCL_UNREACHABLE();
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool cmp_not_equal(_Tp __t, _Up __u) noexcept
{
  return !::cuda::std::cmp_equal(__t, __u);
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool cmp_less(_Tp __t, _Up __u) noexcept
{
  if constexpr (is_signed_v<_Tp> == is_signed_v<_Up>)
  {
    return __t < __u;
  }
  else if constexpr (is_signed_v<_Tp>)
  {
    return __t < 0 ? true : make_unsigned_t<_Tp>(__t) < __u;
  }
  else
  {
    return __u < 0 ? false : __t < make_unsigned_t<_Up>(__u);
  }
  _CCCL_UNREACHABLE();
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool cmp_greater(_Tp __t, _Up __u) noexcept
{
  return ::cuda::std::cmp_less(__u, __t);
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool cmp_less_equal(_Tp __t, _Up __u) noexcept
{
  return !::cuda::std::cmp_greater(__t, __u);
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool cmp_greater_equal(_Tp __t, _Up __u) noexcept
{
  return !::cuda::std::cmp_less(__t, __u);
}

_CCCL_TEMPLATE(class _Tp, class _Up)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp> _CCCL_AND __cccl_is_integer_v<_Up>)
_CCCL_API constexpr bool in_range(_Up __u) noexcept
{
  return ::cuda::std::cmp_less_equal(__u, numeric_limits<_Tp>::max())
      && ::cuda::std::cmp_greater_equal(__u, numeric_limits<_Tp>::min());
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_CMP_H
