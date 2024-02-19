//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COMP_H
#define _LIBCUDACXX___ALGORITHM_COMP_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__type_traits/integral_constant.h"
#if defined(_LIBCUDACXX_HAS_STRING)
#  include "../__type_traits/predicate_traits.h"
#endif // _LIBCUDACXX_HAS_STRING

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __equal_to
{
  template <class _T1, class _T2>
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
  operator()(const _T1& __lhs, const _T2& __rhs) const noexcept(noexcept(__lhs == __rhs))
  {
    return __lhs == __rhs;
  }
};

#if defined(_LIBCUDACXX_HAS_STRING)
template <class _Lhs, class _Rhs>
struct __is_trivial_equality_predicate<__equal_to, _Lhs, _Rhs> : true_type
{};
#endif // _LIBCUDACXX_HAS_STRING

struct __less
{
  template <class _Tp, class _Up>
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
  operator()(const _Tp& __lhs, const _Up& __rhs) const noexcept(noexcept(__lhs < __rhs))
  {
    return __lhs < __rhs;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_COMP_H
