// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_INCREMENTABLE_TRAITS_H
#define _LIBCUDACXX___ITERATOR_INCREMENTABLE_TRAITS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/arithmetic.h"
#include "../__concepts/same_as.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_const.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_pointer.h"
#include "../__type_traits/is_primary_template.h"
#include "../__type_traits/make_signed.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"
#include "../cstddef"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [incrementable.traits]
template<class> struct incrementable_traits {};

template<class _Tp>
requires is_object_v<_Tp>
struct incrementable_traits<_Tp*> {
  using difference_type = ptrdiff_t;
};

template<class _Ip>
struct incrementable_traits<const _Ip> : incrementable_traits<_Ip> {};

template<class _Tp>
concept __has_member_difference_type = requires { typename _Tp::difference_type; };

template<__has_member_difference_type _Tp>
struct incrementable_traits<_Tp> {
  using difference_type = typename _Tp::difference_type;
};

template<class _Tp>
concept __has_integral_minus =
  requires(const _Tp& __x, const _Tp& __y) {
    { __x - __y } -> integral;
  };

template<__has_integral_minus _Tp>
requires (!__has_member_difference_type<_Tp>)
struct incrementable_traits<_Tp> {
  using difference_type = make_signed_t<decltype(declval<_Tp>() - declval<_Tp>())>;
};

template <class>
struct _LIBCUDACXX_TEMPLATE_VIS iterator_traits;

// Let `RI` be `remove_cvref_t<I>`. The type `iter_difference_t<I>` denotes
// `incrementable_traits<RI>::difference_type` if `iterator_traits<RI>` names a specialization
// generated from the primary template, and `iterator_traits<RI>::difference_type` otherwise.
template <class _Ip>
using iter_difference_t = typename conditional_t<__is_primary_template<iterator_traits<remove_cvref_t<_Ip>>>::value,
                                                 incrementable_traits<remove_cvref_t<_Ip> >,
                                                 iterator_traits<remove_cvref_t<_Ip> > >::difference_type;

#elif _LIBCUDACXX_STD_VER > 14

// [incrementable.traits]
template<class, class = void> struct incrementable_traits {};

template<class _Tp>
struct incrementable_traits<_Tp*, enable_if_t<_LIBCUDACXX_TRAIT(is_object, _Tp)>> {
  using difference_type = ptrdiff_t;
};

template<class _Ip>
struct incrementable_traits<const _Ip> : incrementable_traits<_Ip> {};

template<class _Tp, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __has_member_difference_type = false;

template<class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __has_member_difference_type<_Tp, void_t<typename _Tp::difference_type>> = true;

template<class _Tp,class = void, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __has_integral_minus = false;

// In C++17 we get issues trying to bind void* to a const& so special case it here
template<class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __has_integral_minus<_Tp, enable_if_t<!same_as<_Tp, void*>>,
                                                                void_t<decltype(_CUDA_VSTD::declval<const _Tp&>() - _CUDA_VSTD::declval<const _Tp&>())>>
  = integral<decltype(_CUDA_VSTD::declval<const _Tp&>() - _CUDA_VSTD::declval<const _Tp&>())>;

template <class _Tp>
struct incrementable_traits<_Tp, enable_if_t<!_LIBCUDACXX_TRAIT(is_pointer, _Tp) && !_LIBCUDACXX_TRAIT(is_const, _Tp) &&
                                              __has_member_difference_type<_Tp>>> {
  using difference_type = typename _Tp::difference_type;
};

template <class _Tp>
struct incrementable_traits<_Tp, enable_if_t<!_LIBCUDACXX_TRAIT(is_pointer, _Tp) && !_LIBCUDACXX_TRAIT(is_const, _Tp) &&
                                             !__has_member_difference_type<_Tp> && __has_integral_minus<_Tp>>> {
  using difference_type = make_signed_t<decltype(declval<_Tp>() - declval<_Tp>())>;
};

template <class, class = void>
struct _LIBCUDACXX_TEMPLATE_VIS iterator_traits;

// Let `RI` be `remove_cvref_t<I>`. The type `iter_difference_t<I>` denotes
// `incrementable_traits<RI>::difference_type` if `iterator_traits<RI>` names a specialization
// generated from the primary template, and `iterator_traits<RI>::difference_type` otherwise.
template <class _Ip>
using iter_difference_t = typename conditional_t<__is_primary_template<iterator_traits<remove_cvref_t<_Ip>>>::value,
                                                 incrementable_traits<remove_cvref_t<_Ip> >,
                                                 iterator_traits<remove_cvref_t<_Ip> > >::difference_type;

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_INCREMENTABLE_TRAITS_H
