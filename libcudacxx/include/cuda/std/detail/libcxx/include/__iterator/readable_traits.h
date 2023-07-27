// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_READABLE_TRAITS_H
#define _LIBCUDACXX___ITERATOR_READABLE_TRAITS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/is_primary_template.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_const.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_primary_template.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/remove_extent.h"
#include "../__type_traits/void_t.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [readable.traits]
template<class> struct __cond_value_type {};

template<class _Tp>
requires is_object_v<_Tp>
struct __cond_value_type<_Tp> { using value_type = remove_cv_t<_Tp>; };

template<class _Tp>
concept __has_member_value_type = requires { typename _Tp::value_type; };

template<class _Tp>
concept __has_member_element_type = requires { typename _Tp::element_type; };

template<class> struct indirectly_readable_traits {};

template<class _Ip>
requires is_array_v<_Ip>
struct indirectly_readable_traits<_Ip> {
  using value_type = remove_cv_t<remove_extent_t<_Ip>>;
};

template<class _Ip>
struct indirectly_readable_traits<const _Ip> : indirectly_readable_traits<_Ip> {};

template<class _Tp>
struct indirectly_readable_traits<_Tp*> : __cond_value_type<_Tp> {};

template<__has_member_value_type _Tp>
struct indirectly_readable_traits<_Tp>
  : __cond_value_type<typename _Tp::value_type> {};

template<__has_member_element_type _Tp>
struct indirectly_readable_traits<_Tp>
  : __cond_value_type<typename _Tp::element_type> {};

template<__has_member_value_type _Tp>
  requires __has_member_element_type<_Tp>
struct indirectly_readable_traits<_Tp> {};

template<__has_member_value_type _Tp>
  requires __has_member_element_type<_Tp> &&
           same_as<remove_cv_t<typename _Tp::element_type>,
                   remove_cv_t<typename _Tp::value_type>>
struct indirectly_readable_traits<_Tp>
  : __cond_value_type<typename _Tp::value_type> {};

template <class>
struct _LIBCUDACXX_TEMPLATE_VIS iterator_traits;

// Let `RI` be `remove_cvref_t<I>`. The type `iter_value_t<I>` denotes
// `indirectly_readable_traits<RI>::value_type` if `iterator_traits<RI>` names a specialization
// generated from the primary template, and `iterator_traits<RI>::value_type` otherwise.
template <class _Ip>
using iter_value_t = typename conditional_t<__is_primary_template<iterator_traits<remove_cvref_t<_Ip>>>::value,
                                            indirectly_readable_traits<remove_cvref_t<_Ip> >,
                                            iterator_traits<remove_cvref_t<_Ip> > >::value_type;

#elif _LIBCUDACXX_STD_VER > 14

// [readable.traits]
template<class, class = void> struct __cond_value_type {};

template<class _Tp>
struct __cond_value_type<_Tp, enable_if_t<_LIBCUDACXX_TRAIT(is_object, _Tp)>> { using value_type = remove_cv_t<_Tp>; };

template<class _Tp, class = void>
inline constexpr bool __has_member_value_type = false;

template<class _Tp>
inline constexpr bool __has_member_value_type<_Tp, void_t<typename _Tp::value_type>> = true;

template<class _Tp, class = void>
inline constexpr bool __has_member_element_type = false;

template<class _Tp>
inline constexpr bool __has_member_element_type<_Tp, void_t<typename _Tp::element_type>> = true;

template<class, class = void> struct indirectly_readable_traits {};

template <class _Ip>
struct indirectly_readable_traits<_Ip, enable_if_t<!_LIBCUDACXX_TRAIT(is_const,_Ip) && _LIBCUDACXX_TRAIT(is_array, _Ip)>> {
  using value_type = remove_cv_t<remove_extent_t<_Ip>>;
};

template<class _Ip>
struct indirectly_readable_traits<const _Ip>
  : indirectly_readable_traits<_Ip> {};

template<class _Tp>
struct indirectly_readable_traits<_Tp*>
  : __cond_value_type<_Tp> {};

template<class _Tp>
struct indirectly_readable_traits<_Tp, enable_if_t<!_LIBCUDACXX_TRAIT(is_const, _Tp) && __has_member_value_type<_Tp> && !__has_member_element_type<_Tp>>>
  : __cond_value_type<typename _Tp::value_type> {};

template<class _Tp>
struct indirectly_readable_traits<_Tp, enable_if_t<!_LIBCUDACXX_TRAIT(is_const, _Tp) && !__has_member_value_type<_Tp> && __has_member_element_type<_Tp>>>
  : __cond_value_type<typename _Tp::element_type> {};

template<class _Tp>
struct indirectly_readable_traits<_Tp, enable_if_t<!_LIBCUDACXX_TRAIT(is_const, _Tp) && __has_member_value_type<_Tp> && __has_member_element_type<_Tp> &&
                                                   same_as<remove_cv_t<typename _Tp::element_type>,
                                                           remove_cv_t<typename _Tp::value_type>>>>
  : __cond_value_type<typename _Tp::value_type> {};

template <class, class>
struct _LIBCUDACXX_TEMPLATE_VIS iterator_traits;

// Let `RI` be `remove_cvref_t<I>`. The type `iter_value_t<I>` denotes
// `indirectly_readable_traits<RI>::value_type` if `iterator_traits<RI>` names a specialization
// generated from the primary template, and `iterator_traits<RI>::value_type` otherwise.
template <class _Ip>
using iter_value_t = typename conditional_t<__is_primary_template<iterator_traits<remove_cvref_t<_Ip>>>::value,
                                            indirectly_readable_traits<remove_cvref_t<_Ip> >,
                                            iterator_traits<remove_cvref_t<_Ip> > >::value_type;

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_READABLE_TRAITS_H
