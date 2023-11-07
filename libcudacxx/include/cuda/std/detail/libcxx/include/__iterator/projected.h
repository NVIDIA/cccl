// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___ITERATOR_PROJECTED_H
#define _LIBCUDACXX___ITERATOR_PROJECTED_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/remove_cvref.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_TEMPLATE(class _It, class _Proj)
  _LIBCUDACXX_REQUIRES( indirectly_readable<_It> _LIBCUDACXX_AND indirectly_regular_unary_invocable<_Proj, _It>)
struct projected {
  using value_type = remove_cvref_t<indirect_result_t<_Proj&, _It>>;
  _LIBCUDACXX_INLINE_VISIBILITY indirect_result_t<_Proj&, _It> operator*() const; // not defined
};

#if _LIBCUDACXX_STD_VER > 17
template<weakly_incrementable _It, class _Proj>
struct incrementable_traits<projected<_It, _Proj>> {
  using difference_type = iter_difference_t<_It>;
};
#else
template<class _It, class _Proj>
struct incrementable_traits<projected<_It, _Proj>, enable_if_t<weakly_incrementable<_It>>> {
  using difference_type = iter_difference_t<_It>;
};
#endif // _LIBCUDACXX_STD_VER > 17

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_PROJECTED_H
