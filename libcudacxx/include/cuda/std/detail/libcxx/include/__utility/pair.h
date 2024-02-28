//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_PAIR_H
#define _LIBCUDACXX___UTILITY_PAIR_H

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

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include "../__compare/common_comparison_category.h"
#  include "../__compare/synth_three_way.h"
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#include "../__functional/unwrap_ref.h"
#include "../__fwd/get.h"
#include "../__fwd/pair.h"
#include "../__fwd/tuple.h"
#include "../__tuple_dir/sfinae_helpers.h"
#include "../__tuple_dir/structured_bindings.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_indices.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__type_traits/common_reference.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_default_constructible.h"
#include "../__type_traits/is_implicitly_default_constructible.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_nothrow_assignable.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_assignable.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_swappable.h"
#include "../__type_traits/make_const_lvalue_ref.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"
#include "../__utility/piecewise_construct.h"
#include "../cstddef"

// Provide compatability between `std::pair` and `cuda::std::pair`
#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
#  include <utility>
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __invalid_pair_constraints
{
  static constexpr bool __implicit_constructible = false;
  static constexpr bool __explicit_constructible = false;
  static constexpr bool __enable_assign          = false;
};

template <class _T1, class _T2>
struct __pair_constraints
{
  static constexpr bool __implicit_default_constructible =
    __is_implicitly_default_constructible<_T1>::value && __is_implicitly_default_constructible<_T2>::value;

  static constexpr bool __explicit_default_constructible =
    !__implicit_default_constructible && _LIBCUDACXX_TRAIT(is_default_constructible, _T1)
    && _LIBCUDACXX_TRAIT(is_default_constructible, _T2);

  static constexpr bool __explicit_constructible_from_elements =
    _LIBCUDACXX_TRAIT(is_copy_constructible, _T1) && _LIBCUDACXX_TRAIT(is_copy_constructible, _T2)
    && (!_LIBCUDACXX_TRAIT(is_convertible, __make_const_lvalue_ref<_T1>, _T1)
        || !_LIBCUDACXX_TRAIT(is_convertible, __make_const_lvalue_ref<_T2>, _T2));

  static constexpr bool __implicit_constructible_from_elements =
    _LIBCUDACXX_TRAIT(is_copy_constructible, _T1) && _LIBCUDACXX_TRAIT(is_copy_constructible, _T2)
    && _LIBCUDACXX_TRAIT(is_convertible, __make_const_lvalue_ref<_T1>, _T1)
    && _LIBCUDACXX_TRAIT(is_convertible, __make_const_lvalue_ref<_T2>, _T2);

  template <class _U1, class _U2>
  struct __constructible
  {
    static constexpr bool __explicit_constructible =
      _LIBCUDACXX_TRAIT(is_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_constructible, _T2, _U2)
      && (!_LIBCUDACXX_TRAIT(is_convertible, _U1, _T1) || !_LIBCUDACXX_TRAIT(is_convertible, _U2, _T2));

    static constexpr bool __implicit_constructible =
      _LIBCUDACXX_TRAIT(is_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_constructible, _T2, _U2)
      && _LIBCUDACXX_TRAIT(is_convertible, _U1, _T1) && _LIBCUDACXX_TRAIT(is_convertible, _U2, _T2);
  };

  template <class _U1, class _U2>
  struct __assignable
  {
    static constexpr bool __enable_assign =
      _LIBCUDACXX_TRAIT(is_assignable, _T1&, _U1) && _LIBCUDACXX_TRAIT(is_assignable, _T2&, _U2);
  };
};

// We need to synthesize the copy / move assignment if it would be implicitly deleted as a member of a class
// In that case _T1 would be copy assignable but _TestSynthesizeAssignment<_T1> would not
// This happens e.g for reference types
template <class _T1>
struct _TestSynthesizeAssignment
{
  _T1 __dummy;
};

template <class _T1, class _T2>
struct __must_synthesize_assignment
    : integral_constant<bool,
                        (_LIBCUDACXX_TRAIT(is_copy_assignable, _T1) && _LIBCUDACXX_TRAIT(is_copy_assignable, _T2)
                         && !(_LIBCUDACXX_TRAIT(is_copy_assignable, _TestSynthesizeAssignment<_T1>)
                              && _LIBCUDACXX_TRAIT(is_copy_assignable, _TestSynthesizeAssignment<_T2>)))
                          || (_LIBCUDACXX_TRAIT(is_move_assignable, _T1) && _LIBCUDACXX_TRAIT(is_move_assignable, _T2)
                              && !(_LIBCUDACXX_TRAIT(is_move_assignable, _TestSynthesizeAssignment<_T1>)
                                   && _LIBCUDACXX_TRAIT(is_move_assignable, _TestSynthesizeAssignment<_T2>)))>
{};

// base class to ensure `is_trivially_copyable` when possible
template <class _T1, class _T2, bool = __must_synthesize_assignment<_T1, _T2>::value>
struct __pair_base
{
  _T1 first;
  _T2 second;

  template <class _Constraints                                                 = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit constexpr __pair_base() noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T1)
    && _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  template <class _Constraints                                                 = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr __pair_base() noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T1)
    && _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  template <class _U1, class _U2>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr __pair_base(_U1&& __t1, _U2&& __t2) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : first(_CUDA_VSTD::forward<_U1>(__t1))
      , second(_CUDA_VSTD::forward<_U2>(__t2))
  {}

protected:
  template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __pair_base(
    piecewise_construct_t,
    tuple<_Args1...>& __first_args,
    tuple<_Args2...>& __second_args,
    __tuple_indices<_I1...>,
    __tuple_indices<_I2...>);
};

template <class _T1, class _T2>
struct __pair_base<_T1, _T2, true>
{
  _T1 first;
  _T2 second;

  template <class _Constraints                                                 = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit constexpr __pair_base() noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T1)
    && _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  template <class _Constraints                                                 = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr __pair_base() noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T1)
    && _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  constexpr __pair_base(const __pair_base&) = default;
  constexpr __pair_base(__pair_base&&)      = default;

  // We need to ensure that a reference type, which would inhibit the implicit copy assignment still works
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __pair_base& operator=(
    __conditional_t<_LIBCUDACXX_TRAIT(is_copy_assignable, _T1) && _LIBCUDACXX_TRAIT(is_copy_assignable, _T2),
                    __pair_base,
                    __nat> const& __p) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _T1)
                                                && _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _T2))
  {
    first  = __p.first;
    second = __p.second;
    return *this;
  }

  // We need to ensure that a reference type, which would inhibit the implicit move assignment still works
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __pair_base& operator=(
    __conditional_t<_LIBCUDACXX_TRAIT(is_move_assignable, _T1) && _LIBCUDACXX_TRAIT(is_move_assignable, _T2),
                    __pair_base,
                    __nat>&& __p) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_assignable, _T1)
                                           && _LIBCUDACXX_TRAIT(is_nothrow_move_assignable, _T2))
  {
    first  = _CUDA_VSTD::forward<_T1>(__p.first);
    second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }

  template <class _U1, class _U2>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr __pair_base(_U1&& __t1, _U2&& __t2) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : first(_CUDA_VSTD::forward<_U1>(__t1))
      , second(_CUDA_VSTD::forward<_U2>(__t2))
  {}

protected:
  template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __pair_base(
    piecewise_construct_t,
    tuple<_Args1...>& __first_args,
    tuple<_Args2...>& __second_args,
    __tuple_indices<_I1...>,
    __tuple_indices<_I2...>);
};

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS pair : public __pair_base<_T1, _T2>
{
  using __base = __pair_base<_T1, _T2>;

  typedef _T1 first_type;
  typedef _T2 second_type;

  template <class _Constraints                                                 = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit constexpr pair() noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T1)
    && _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T2))
      : __base()
  {}

  template <class _Constraints                                                 = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr pair() noexcept(_LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T1)
                                                          && _LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _T2))
      : __base()
  {}

  // element wise constructors
  template <class _Constraints                                                     = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__explicit_constructible_from_elements, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit constexpr pair(const _T1& __t1, const _T2& __t2) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _T1) && _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _T2))
      : __base(__t1, __t2)
  {}

  template <class _Constraints                                                     = __pair_constraints<_T1, _T2>,
            __enable_if_t<_Constraints::__implicit_constructible_from_elements, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr pair(const _T1& __t1, const _T2& __t2) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _T1) && _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _T2))
      : __base(__t1, __t2)
  {}

  template <class _U1          = _T1,
            class _U2          = _T2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>,
            __enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit constexpr pair(_U1&& __u1, _U2&& __u2) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__u1), _CUDA_VSTD::forward<_U2>(__u2))
  {}

  template <class _U1          = _T1,
            class _U2          = _T2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>,
            __enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr pair(_U1&& __u1, _U2&& __u2) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__u1), _CUDA_VSTD::forward<_U2>(__u2))
  {}

  template <class... _Args1, class... _Args2>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  pair(piecewise_construct_t __pc, tuple<_Args1...> __first_args, tuple<_Args2...> __second_args) noexcept(
    (is_nothrow_constructible<_T1, _Args1...>::value && is_nothrow_constructible<_T2, _Args2...>::value))
      : __base(__pc,
               __first_args,
               __second_args,
               __make_tuple_indices_t<sizeof...(_Args1)>(),
               __make_tuple_indices_t<sizeof...(_Args2)>())
  {}

  // copy and move constructors
  pair(pair const&) = default;
  pair(pair&&)      = default;

  template <class _U1 = _T1,
            class _U2 = _T2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<const _U1&, const _U2&>,
            __enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair(const pair<_U1, _U2>& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, const _U1&)
    && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  template <class _U1 = _T1,
            class _U2 = _T2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<const _U1&, const _U2&>,
            __enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair(const pair<_U1, _U2>& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, const _U1&)
    && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  // move constructors
  template <class _U1          = _T1,
            class _U2          = _T2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>,
            __enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY explicit _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair(pair<_U1, _U2>&& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}

  template <class _U1          = _T1,
            class _U2          = _T2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>,
            __enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair(pair<_U1, _U2>&& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}

  // std compatability
#if defined(__cuda_std__) && !defined(_CCCL_COMPILER_NVRTC)
  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<const _U1&, const _U2&>,
            __enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _LIBCUDACXX_HOST _LIBCUDACXX_HIDE_FROM_ABI explicit _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair(const ::std::pair<_U1, _U2>& __p) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, const _U1&)
                                                  && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<const _U1&, const _U2&>,
            __enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _LIBCUDACXX_HOST _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair(const ::std::pair<_U1, _U2>& __p) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, const _U1&)
                                                  && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>,
            __enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _LIBCUDACXX_HOST _LIBCUDACXX_HIDE_FROM_ABI explicit _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair(::std::pair<_U1, _U2>&& __p) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1)
                                             && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>,
            __enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _LIBCUDACXX_HOST _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair(::std::pair<_U1, _U2>&& __p) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _U1)
                                             && _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}
#endif // defined(__cuda_std__) && !defined(_CCCL_COMPILER_NVRTC)

  // assignments
  pair& operator=(const pair&) = default;
  pair& operator=(pair&&)      = default;

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __assignable<const _U1&, const _U2&>,
            __enable_if_t<_Constraints::__enable_assign, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair& operator=(const pair<_U1, _U2>& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_assignable, _T1, const _U1&)
    && _LIBCUDACXX_TRAIT(is_nothrow_assignable, _T2, const _U2&))
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __assignable<_U1, _U2>,
            __enable_if_t<_Constraints::__enable_assign, int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair& operator=(pair<_U1, _U2>&& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_assignable, _T1, _U1) && _LIBCUDACXX_TRAIT(is_nothrow_assignable, _T2, _U2))
  {
    this->first  = _CUDA_VSTD::forward<_U1>(__p.first);
    this->second = _CUDA_VSTD::forward<_U2>(__p.second);
    return *this;
  }

  // std assignments
#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
  template <class _UT1 = _T1, __enable_if_t<is_copy_assignable<_UT1>::value && is_copy_assignable<_T2>::value, int> = 0>
  _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair& operator=(::std::pair<_T1, _T2> const& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _T1) && _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _T2))
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

  template <class _UT1 = _T1, __enable_if_t<is_move_assignable<_UT1>::value && is_move_assignable<_T2>::value, int> = 0>
  _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 pair& operator=(::std::pair<_T1, _T2>&& __p) noexcept(
    _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _T1) && _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _T2))
  {
    this->first  = _CUDA_VSTD::forward<_T1>(__p.first);
    this->second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

#if _CCCL_STD_VER >= 2023
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const pair& operator=(pair const& __p) const
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, const _T1)
             && _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, const _T2))
    requires(is_copy_assignable_v<const _T1> && is_copy_assignable_v<const _T2>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

#  if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr const pair& operator=(::std::pair<_T1, _T2> const& __p) const
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, const _T1)
             && _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, const _T2))
    requires(is_copy_assignable_v<const _T1> && is_copy_assignable_v<const _T2>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }
#  endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const pair& operator=(pair&& __p) const
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_assignable, const _T1&, _T1)
             && _LIBCUDACXX_TRAIT(is_nothrow_assignable, const _T2&, _T2))
    requires(is_assignable_v<const _T1&, _T1> && is_assignable_v<const _T2&, _T2>)
  {
    this->first  = _CUDA_VSTD::forward<_T1>(__p.first);
    this->second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }

#  if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr const pair& operator=(::std::pair<_T1, _T2>&& __p) const
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_assignable, const _T1&, _T1)
             && _LIBCUDACXX_TRAIT(is_nothrow_assignable, const _T2&, _T2))
    requires(is_assignable_v<const _T1&, _T1> && is_assignable_v<const _T2&, _T2>)
  {
    this->first  = _CUDA_VSTD::forward<_T1>(__p.first);
    this->second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }
#  endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

  template <class _U1, class _U2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const pair&
  operator=(const pair<_U1, _U2>& __p) const
    requires(is_assignable_v<const _T1&, const _U1&> && is_assignable_v<const _T2&, const _U2&>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

#  if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
  template <class _U1, class _U2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr const pair& operator=(const ::std::pair<_U1, _U2>& __p) const
    requires(is_assignable_v<const _T1&, const _U1&> && is_assignable_v<const _T2&, const _U2&>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }
#  endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)

  template <class _U1, class _U2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const pair& operator=(pair<_U1, _U2>&& __p) const
    requires(is_assignable_v<const _T1&, _U1> && is_assignable_v<const _T2&, _U2>)
  {
    this->first  = _CUDA_VSTD::forward<_U1>(__p.first);
    this->second = _CUDA_VSTD::forward<_U2>(__p.second);
    return *this;
  }

#  if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
  template <class _U1, class _U2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_HOST constexpr const pair& operator=(::std::pair<_U1, _U2>&& __p) const
    requires(is_assignable_v<const _T1&, _U1> && is_assignable_v<const _T2&, _U2>)
  {
    this->first  = _CUDA_VSTD::forward<_U1>(__p.first);
    this->second = _CUDA_VSTD::forward<_U2>(__p.second);
    return *this;
  }
#  endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)
#endif // _CCCL_STD_VER >= 2023

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 void
  swap(pair& __p) noexcept(__is_nothrow_swappable<_T1>::value && __is_nothrow_swappable<_T2>::value)
  {
    using _CUDA_VSTD::swap;
    swap(this->first, __p.first);
    swap(this->second, __p.second);
  }

#if _CCCL_STD_VER >= 2023
  _LIBCUDACXX_HIDE_FROM_ABI constexpr void swap(const pair& __p) const
    noexcept(__is_nothrow_swappable<const _T1>::value && __is_nothrow_swappable<const _T2>::value)
  {
    using _CUDA_VSTD::swap;
    swap(this->first, __p.first);
    swap(this->second, __p.second);
  }
#endif // _CCCL_STD_VER >= 2023

#if defined(__cuda_std__) && !defined(__CUDACC_RTC__)
  _LIBCUDACXX_HOST _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 operator ::std::pair<_T1, _T2>() const
  {
    return {this->first, this->second};
  }
#endif // defined(__cuda_std__) && !defined(__CUDACC_RTC__)
};

#if _CCCL_STD_VER > 2014 && !defined(_LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES)
template <class _T1, class _T2>
_LIBCUDACXX_HOST_DEVICE pair(_T1, _T2) -> pair<_T1, _T2>;
#endif // _CCCL_STD_VER > 2014 && !defined(_LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES)

// [pairs.spec], specialized algorithms

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator==(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __x.first == __y.first && __x.second == __y.second;
}

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

template <class _T1, class _T2>
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr common_comparison_category_t<
  __synth_three_way_result<_T1>,
  __synth_three_way_result<_T2> >
operator<=>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  if (auto __c = _CUDA_VSTD::__synth_three_way(__x.first, __y.first); __c != 0)
  {
    return __c;
  }
  return _CUDA_VSTD::__synth_three_way(__x.second, __y.second);
}

#else // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator!=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__x == __y);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator<(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __y < __x;
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator>=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__x < __y);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 bool
operator<=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__y < __x);
}

#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#if _CCCL_STD_VER >= 2023
template <class _T1, class _T2, class _U1, class _U2, template <class> class _TQual, template <class> class _UQual>
  requires requires {
    typename pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>, common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
  }
struct basic_common_reference<pair<_T1, _T2>, pair<_U1, _U2>, _TQual, _UQual>
{
  using type = pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>, common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
};

template <class _T1, class _T2, class _U1, class _U2>
  requires requires { typename pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>; }
struct common_type<pair<_T1, _T2>, pair<_U1, _U2>>
{
  using type = pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>;
};
#endif // _CCCL_STD_VER >= 2023

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<__is_swappable<_T1>::value && __is_swappable<_T2>::value, void>
  swap(pair<_T1, _T2>& __x,
       pair<_T1, _T2>& __y) noexcept((__is_nothrow_swappable<_T1>::value && __is_nothrow_swappable<_T2>::value))
{
  __x.swap(__y);
}

#if _CCCL_STD_VER >= 2023
template <class _T1, class _T2>
  requires(__is_swappable<const _T1>::value && __is_swappable<const _T2>::value)
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr void
swap(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y) noexcept(noexcept(__x.swap(__y)))
{
  __x.swap(__y);
}
#endif // _CCCL_STD_VER >= 2023

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
  pair<typename __unwrap_ref_decay<_T1>::type, typename __unwrap_ref_decay<_T2>::type>
  make_pair(_T1&& __t1, _T2&& __t2)
{
  return pair<typename __unwrap_ref_decay<_T1>::type, typename __unwrap_ref_decay<_T2>::type>(
    _CUDA_VSTD::forward<_T1>(__t1), _CUDA_VSTD::forward<_T2>(__t2));
}

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_size<pair<_T1, _T2> > : public integral_constant<size_t, 2>
{};

template <size_t _Ip, class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, pair<_T1, _T2> >
{
  static_assert(_Ip < 2, "Index out of bounds in std::tuple_element<std::pair<T1, T2>>");
};

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<0, pair<_T1, _T2> >
{
  typedef _LIBCUDACXX_NODEBUG_TYPE _T1 type;
};

template <class _T1, class _T2>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<1, pair<_T1, _T2> >
{
  typedef _LIBCUDACXX_NODEBUG_TYPE _T2 type;
};

template <size_t _Ip>
struct __get_pair;

template <>
struct __get_pair<0>
{
  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _T1& get(pair<_T1, _T2>& __p) noexcept
  {
    return __p.first;
  }

  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const _T1&
  get(const pair<_T1, _T2>& __p) noexcept
  {
    return __p.first;
  }

  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _T1&& get(pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<_T1>(__p.first);
  }

  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const _T1&&
  get(const pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<const _T1>(__p.first);
  }
};

template <>
struct __get_pair<1>
{
  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _T2& get(pair<_T1, _T2>& __p) noexcept
  {
    return __p.second;
  }

  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const _T2&
  get(const pair<_T1, _T2>& __p) noexcept
  {
    return __p.second;
  }

  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _T2&& get(pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<_T2>(__p.second);
  }

  template <class _T1, class _T2>
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const _T2&&
  get(const pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<const _T2>(__p.second);
  }
};

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __tuple_element_t<_Ip, pair<_T1, _T2>>&
get(pair<_T1, _T2>& __p) noexcept
{
  return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const __tuple_element_t<_Ip, pair<_T1, _T2>>&
get(const pair<_T1, _T2>& __p) noexcept
{
  return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __tuple_element_t<_Ip, pair<_T1, _T2>>&&
get(pair<_T1, _T2>&& __p) noexcept
{
  return __get_pair<_Ip>::get(_CUDA_VSTD::move(__p));
}

template <size_t _Ip, class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const __tuple_element_t<_Ip, pair<_T1, _T2>>&&
get(const pair<_T1, _T2>&& __p) noexcept
{
  return __get_pair<_Ip>::get(_CUDA_VSTD::move(__p));
}

#if _CCCL_STD_VER >= 2014
template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1& get(pair<_T1, _T2>& __p) noexcept
{
  return __get_pair<0>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1 const& get(pair<_T1, _T2> const& __p) noexcept
{
  return __get_pair<0>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1&& get(pair<_T1, _T2>&& __p) noexcept
{
  return __get_pair<0>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1 const&& get(pair<_T1, _T2> const&& __p) noexcept
{
  return __get_pair<0>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1& get(pair<_T2, _T1>& __p) noexcept
{
  return __get_pair<1>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1 const& get(pair<_T2, _T1> const& __p) noexcept
{
  return __get_pair<1>::get(__p);
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1&& get(pair<_T2, _T1>&& __p) noexcept
{
  return __get_pair<1>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr _T1 const&& get(pair<_T2, _T1> const&& __p) noexcept
{
  return __get_pair<1>::get(_CUDA_VSTD::move(__p));
}

#endif // _CCCL_STD_VER >= 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_PAIR_H
