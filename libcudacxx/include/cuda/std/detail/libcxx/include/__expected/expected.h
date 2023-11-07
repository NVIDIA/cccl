//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___EXPECTED_EXPECTED_H
#define _LIBCUDACXX___EXPECTED_EXPECTED_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__concepts/invocable.h"
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
#include "../__expected/bad_expected_access.h"
#endif // _LIBCUDACXX_NO_EXCEPTIONS
#include "../__expected/expected_base.h"
#include "../__expected/unexpect.h"
#include "../__expected/unexpected.h"
#include "../__memory/addressof.h"
#include "../__memory/construct_at.h"
#include "../__type_traits/conjunction.h"
#include "../__type_traits/disjunction.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__type_traits/is_default_constructible.h"
#include "../__type_traits/is_function.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_assignable.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_swappable.h"
#include "../__type_traits/is_trivially_copy_constructible.h"
#include "../__type_traits/is_trivially_destructible.h"
#include "../__type_traits/is_trivially_move_constructible.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/lazy.h"
#include "../__type_traits/negation.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/exception_guard.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"
#include "../__utility/swap.h"

#include "../cstdlib"
#include "../initializer_list"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, class _Err>
class expected;

namespace __expected {

  template <class _Err, class _Arg>
  _LIBCUDACXX_INLINE_VISIBILITY
  void __throw_bad_expected_access(_Arg&& __arg) {
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
    throw _CUDA_STD::bad_expected_access<_Err>(_CUDA_VSTD::forward<_Arg>(__arg));
#else
    (void)__arg;
    _LIBCUDACXX_UNREACHABLE();
#endif // _LIBCUDACXX_NO_EXCEPTIONS
  }

  template <class _Tp, class _Err>
  _LIBCUDACXX_INLINE_VAR constexpr bool __valid_expected =
    !_LIBCUDACXX_TRAIT(is_reference, _Tp) &&
    !_LIBCUDACXX_TRAIT(is_function, _Tp) &&
    !_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<_Tp>, in_place_t) &&
    !_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<_Tp>, unexpect_t) &&
    !__unexpected::__is_unexpected<__remove_cv_t<_Tp>> &&
    __unexpected::__valid_unexpected<_Err>;

  template <class _Tp>
  _LIBCUDACXX_INLINE_VAR constexpr bool __is_expected = false;

  template <class _Tp, class _Err>
  _LIBCUDACXX_INLINE_VAR constexpr bool __is_expected<expected<_Tp, _Err>> = true;

  template <class _Tp>
  _LIBCUDACXX_INLINE_VAR constexpr bool __is_expected_nonvoid = __is_expected<_Tp>;

  template <class _Err>
  _LIBCUDACXX_INLINE_VAR constexpr bool __is_expected_nonvoid<expected<void, _Err>> = false;

  template<class _Tp, class _Err>
  _LIBCUDACXX_INLINE_VAR constexpr bool __can_swap =
       _LIBCUDACXX_TRAIT(is_swappable, _Tp)
    && _LIBCUDACXX_TRAIT(is_swappable, _Err)
    && _LIBCUDACXX_TRAIT(is_move_constructible, _Tp)
    && _LIBCUDACXX_TRAIT(is_move_constructible, _Err)
    && (_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp)
    || _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err));

  template<class _Err>
  _LIBCUDACXX_INLINE_VAR constexpr bool __can_swap<void, _Err> =
       _LIBCUDACXX_TRAIT(is_swappable, _Err)
    && _LIBCUDACXX_TRAIT(is_move_constructible, _Err);
} // namespace __expected

template <class _Tp, class _Err>
class expected : private __expected_move_assign<_Tp, _Err>
               , private __expected_sfinae_ctor_base_t<_Tp, _Err>
               , private __expected_sfinae_assign_base_t<_Tp, _Err>
{
  using __base = __expected_move_assign<_Tp, _Err>;

  static_assert(__expected::__valid_expected<_Tp, _Err>,
      "[expected.object.general] A program that instantiates the definition of template expected<T, E> for a "
      "reference type, a function type, or for possibly cv-qualified types in_place_t, unexpect_t, or a "
      "specialization of unexpected for the T parameter is ill-formed. A program that instantiates the "
      "definition of the template expected<T, E> with a type for the E parameter that is not a valid "
      "template argument for unexpected is ill-formed.");

  template <class, class>
  friend class expected;

public:
  using value_type      = _Tp;
  using error_type      = _Err;
  using unexpected_type = unexpected<_Err>;

  template <class _Up>
  using rebind = expected<_Up, error_type>;

  // [expected.object.ctor], constructors
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_default_constructible, _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected() noexcept(_LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _Tp2))
    : __base(true)
  {}

  constexpr expected(const expected&) = default;
  constexpr expected(expected&&) = default;
  constexpr expected& operator=(const expected&) = default;
  constexpr expected& operator=(expected&&) = default;

private:
  template <class _Up, class _OtherErr, class _UfQual, class _OtherErrQual>
  using __can_convert =
      _And< is_constructible<_Tp, _UfQual>,
            is_constructible<_Err, _OtherErrQual>,
            _Not<is_constructible<_Tp, expected<_Up, _OtherErr>&>>,
            _Not<is_constructible<_Tp, expected<_Up, _OtherErr>>>,
            _Not<is_constructible<_Tp, const expected<_Up, _OtherErr>&>>,
            _Not<is_constructible<_Tp, const expected<_Up, _OtherErr>>>,
            _Not<is_convertible<expected<_Up, _OtherErr>&, _Tp>>,
            _Not<is_convertible<expected<_Up, _OtherErr>&&, _Tp>>,
            _Not<is_convertible<const expected<_Up, _OtherErr>&, _Tp>>,
            _Not<is_convertible<const expected<_Up, _OtherErr>&&, _Tp>>,
            _Not<is_constructible<unexpected<_Err>, expected<_Up, _OtherErr>&>>,
            _Not<is_constructible<unexpected<_Err>, expected<_Up, _OtherErr>>>,
            _Not<is_constructible<unexpected<_Err>, const expected<_Up, _OtherErr>&>>,
            _Not<is_constructible<unexpected<_Err>, const expected<_Up, _OtherErr>>> >;


public:
  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, const _Up&, const _OtherErr&>::value _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, const _Up&, _Tp) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected(const expected<_Up, _OtherErr>& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, const _Up&) &&
             _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(__other.__has_val_)
  {
    if (__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, __other.__union_.__val_);
    } else {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, const _Up&, const _OtherErr&>::value _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_convertible, const _Up&, _Tp) || !_LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  explicit expected(const expected<_Up, _OtherErr>& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, const _Up&) &&
             _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(__other.__has_val_)
  {
    if (__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, __other.__union_.__val_);
    } else {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, _Up, _OtherErr>::value _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, _Up, _Tp) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected(expected<_Up, _OtherErr>&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Up) &&
             _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(__other.__has_val_)
  {
    if (__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, _CUDA_VSTD::move(__other.__union_.__val_));
    } else {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, _Up, _OtherErr>::value _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_convertible, _Up, _Tp) || !_LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  explicit expected(expected<_Up, _OtherErr>&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Up) &&
             _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(__other.__has_val_)
  {
    if (__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, _CUDA_VSTD::move(__other.__union_.__val_));
    } else {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up = _Tp)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_same, __remove_cvref_t<_Up>, in_place_t)) _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_same, expected, __remove_cvref_t<_Up>)) _LIBCUDACXX_AND
              (!__unexpected::__is_unexpected<__remove_cvref_t<_Up>>) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_constructible, _Tp, _Up) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, _Up, _Tp)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(_Up&& __u) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Up)) // strengthened
    : __base(in_place, _CUDA_VSTD::forward<_Up>(__u))
  {}

  _LIBCUDACXX_TEMPLATE(class _Up = _Tp)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_same, __remove_cvref_t<_Up>, in_place_t)) _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_same, expected, __remove_cvref_t<_Up>)) _LIBCUDACXX_AND
              (!__unexpected::__is_unexpected<__remove_cvref_t<_Up>>) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_constructible, _Tp, _Up) _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_convertible, _Up, _Tp))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(_Up&& __u) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Up)) // strengthened
    : __base(in_place, _CUDA_VSTD::forward<_Up>(__u))
  {}

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, const _OtherErr&) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(const unexpected<_OtherErr>& __unex) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(unexpect, __unex.error())
  {}

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, const _OtherErr&) _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(const unexpected<_OtherErr>& __unex) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(unexpect, __unex.error())
  {}

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _OtherErr) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(unexpected<_OtherErr>&& __unex) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(unexpect, _CUDA_VSTD::move(__unex.error()))
  {}


  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _OtherErr) _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(unexpected<_OtherErr>&& __unex) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(unexpect, _CUDA_VSTD::move(__unex.error()))
  {}

  _LIBCUDACXX_TEMPLATE(class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Tp, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(in_place_t, _Args&&... __args) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...)) // strengthened
    : __base(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _LIBCUDACXX_TEMPLATE(class _Up, class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Tp, initializer_list<_Up>&, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(in_place_t, initializer_list<_Up> __il, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, initializer_list<_Up>&, _Args...)) // strengthened
    : __base(in_place, __il, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _LIBCUDACXX_TEMPLATE(class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(unexpect_t, _Args&&... __args) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))  // strengthened
    : __base(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _LIBCUDACXX_TEMPLATE(class _Up, class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, initializer_list<_Up>&, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(unexpect_t, initializer_list<_Up> __il, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, initializer_list<_Up>&, _Args...)) // strengthened
    : __base(unexpect, __il, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

private:
  template<class _Fun, class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __base(__expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __base(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
  {}

public:
  // [expected.object.assign], assignment
  _LIBCUDACXX_TEMPLATE(class _Up = _Tp)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_same, expected, __remove_cvref_t<_Up>)) _LIBCUDACXX_AND
              (!__unexpected::__is_unexpected<__remove_cvref_t<_Up>>) _LIBCUDACXX_AND
               _LIBCUDACXX_TRAIT(is_constructible, _Tp, _Up) _LIBCUDACXX_AND
               _LIBCUDACXX_TRAIT(is_assignable, _Tp&, _Up) _LIBCUDACXX_AND
              (_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Up) ||
               _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp) ||
               _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected& operator=(_Up&& __v) {
    if (this->__has_val_) {
      this->__union_.__val_ = _CUDA_VSTD::forward<_Up>(__v);
    } else {
      this->__reinit_expected(this->__union_.__val_, this->__union_.__unex_, _CUDA_VSTD::forward<_Up>(__v));
      this->__has_val_ = true;
    }
    return *this;
  }

private:
  template <class _OtherErrQual>
  static constexpr bool __can_assign_from_unexpected =
      _And< is_constructible<_Err, _OtherErrQual>,
            is_assignable<_Err&, _OtherErrQual>,
            _Lazy<_Or,
                  is_nothrow_constructible<_Err, _OtherErrQual>,
                  is_nothrow_move_constructible<_Tp>,
                  is_nothrow_move_constructible<_Err>> >::value;

public:
  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_assign_from_unexpected<const _OtherErr&>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected& operator=(const unexpected<_OtherErr>& __un) {
    if (this->__has_val_) {
      this->__reinit_expected(this->__union_.__unex_, this->__union_.__val_, __un.error());
      this->__has_val_ = false;
    } else {
      this->__union_.__unex_ = __un.error();
    }
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_assign_from_unexpected<_OtherErr>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected& operator=(unexpected<_OtherErr>&& __un) {
    if (this->__has_val_) {
      this->__reinit_expected(this->__union_.__unex_, this->__union_.__val_, _CUDA_VSTD::move(__un.error()));
      this->__has_val_ = false;
    } else {
      this->__union_.__unex_ = _CUDA_VSTD::move(__un.error());
    }
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  _Tp& emplace(_Args&&... __args) noexcept {
    if (this->__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__val_));
    } else {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__unex_));
      this->__has_val_ = true;
    }
    return *_LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, _CUDA_VSTD::forward<_Args>(__args)...);
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, initializer_list<_Up>&, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  _Tp& emplace(initializer_list<_Up> __il, _Args&&... __args) noexcept {
    if (this->__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__val_));
    } else {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__unex_));
      this->__has_val_ = true;
    }
    return *_LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, __il, _CUDA_VSTD::forward<_Args>(__args)...);
  }


public:
  // [expected.object.swap], swap
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( __expected::__can_swap<_Tp2, _Err2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void swap(expected<_Tp2, _Err>& __rhs)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp2) &&
             _LIBCUDACXX_TRAIT(is_nothrow_swappable, _Tp2) &&
             _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err) &&
             _LIBCUDACXX_TRAIT(is_nothrow_swappable, _Err))
  {
    if (this->__has_val_) {
      if (__rhs.__has_val_) {
        using _CUDA_VSTD::swap;
        swap(this->__union_.__val_, __rhs.__union_.__val_);
      } else {
        this->__swap_val_unex_impl(*this, __rhs);
      }
    } else {
      if (__rhs.__has_val_) {
        this->__swap_val_unex_impl(__rhs, *this);
      } else {
        using _CUDA_VSTD::swap;
        swap(this->__union_.__unex_, __rhs.__union_.__unex_);
      }
    }
  }

  template<class _Tp2 = _Tp, class _Err2 = _Err>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  auto swap(expected& __x, expected& __y) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp2) &&
                                                   _LIBCUDACXX_TRAIT(is_nothrow_swappable, _Tp2) &&
                                                   _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err2) &&
                                                   _LIBCUDACXX_TRAIT(is_nothrow_swappable, _Err2))
    _LIBCUDACXX_TRAILING_REQUIRES(void)(__expected::__can_swap<_Tp2, _Err2>)
  {
    return __x.swap(__y); // some compiler warn about non void function without return
  }

  // [expected.object.obs], observers
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp* operator->() const noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator-> requires the expected to contain a value");
    return _CUDA_VSTD::addressof(this->__union_.__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp* operator->() noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator-> requires the expected to contain a value");
    return _CUDA_VSTD::addressof(this->__union_.__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp& operator*() const& noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator* requires the expected to contain a value");
    return this->__union_.__val_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp& operator*() & noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator* requires the expected to contain a value");
    return this->__union_.__val_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp&& operator*() const&& noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator* requires the expected to contain a value");
    return _CUDA_VSTD::move(this->__union_.__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp&& operator*() && noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator* requires the expected to contain a value");
    return _CUDA_VSTD::move(this->__union_.__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit operator bool() const noexcept { return this->__has_val_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool has_value() const noexcept { return this->__has_val_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Tp& value() const& {
    if (!this->__has_val_) {
      __expected::__throw_bad_expected_access<_Err>(this->__union_.__unex_);
    }
    return this->__union_.__val_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Tp& value() & {
    if (!this->__has_val_) {
      __expected::__throw_bad_expected_access<_Err>(this->__union_.__unex_);
    }
    return this->__union_.__val_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  const _Tp&& value() const&& {
    if (!this->__has_val_) {
      __expected::__throw_bad_expected_access<_Err>(_CUDA_VSTD::move(this->__union_.__unex_));
    }
    return _CUDA_VSTD::move(this->__union_.__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Tp&& value() && {
    if (!this->__has_val_) {
      __expected::__throw_bad_expected_access<_Err>(_CUDA_VSTD::move(this->__union_.__unex_));
    }
    return _CUDA_VSTD::move(this->__union_.__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  const _Err& error() const& noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return this->__union_.__unex_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Err& error() & noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return this->__union_.__unex_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  const _Err&& error() const&& noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return _CUDA_VSTD::move(this->__union_.__unex_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Err&& error() && noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return _CUDA_VSTD::move(this->__union_.__unex_);
  }

  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Tp value_or(_Up&& __v) const& {
    static_assert(_LIBCUDACXX_TRAIT(is_copy_constructible, _Tp), "value_type has to be copy constructible");
    static_assert(_LIBCUDACXX_TRAIT(is_convertible, _Up, _Tp), "argument has to be convertible to value_type");
    return this->__has_val_ ? this->__union_.__val_ : static_cast<_Tp>(_CUDA_VSTD::forward<_Up>(__v));
  }

  template <class _Up>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  _Tp value_or(_Up&& __v) && {
    static_assert(_LIBCUDACXX_TRAIT(is_move_constructible, _Tp), "value_type has to be move constructible");
    static_assert(_LIBCUDACXX_TRAIT(is_convertible, _Up, _Tp), "argument has to be convertible to value_type");
    return this->__has_val_ ? _CUDA_VSTD::move(this->__union_.__val_) : static_cast<_Tp>(_CUDA_VSTD::forward<_Up>(__v));
  }

  // [expected.object.monadic]
  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, _Err2&))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) & {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, _Tp&>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__val_);
    } else {
      return _Res{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Err2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) const& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, const _Tp&>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__val_);
    } else {
      return _Res{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Err2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) && {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, _Tp>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__val_));
    } else {
      return _Res{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, const _Err2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) const&& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, const _Tp>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__val_));
    } else {
      return _Res{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Tp2, _Tp2&))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) & {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, _Err&>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, _Tp),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{in_place, this->__union_.__val_};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) const& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, const _Err&>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, _Tp),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{in_place, this->__union_.__val_};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) && {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, _Err>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, _Tp),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{in_place, _CUDA_VSTD::move(this->__union_.__val_)};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Tp2, const _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) const&& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, const _Err>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, _Tp),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{in_place, _CUDA_VSTD::move(this->__union_.__val_)};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, _Err2&) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, _Tp2&>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) & {
    static_assert(invocable<_Fun, _Tp&>,
      "std::expected::transform requires that F must be invocable with T.");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Tp&>>;

    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__val_);
      return expected<void, _Err>{};
    } else {
      return expected<_Res, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, _Err2&) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, _Tp2&>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) & {
    static_assert(invocable<_Fun, _Tp&>,
      "std::expected::transform requires that F must be invocable with T.");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Tp&>>;

    static_assert(__invoke_constructible<_Fun, _Tp&>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__val_};
    } else {
      return expected<_Res, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Err2) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, const _Tp2&>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const& {
    static_assert(invocable<_Fun, const _Tp&>,
      "std::expected::transform requires that F must be invocable with T.");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Tp&>>;

    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__val_);
      return expected<_Res, _Err>{};
    } else {
      return expected<_Res, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Err2) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, const _Tp2&>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const& {
    static_assert(invocable<_Fun, const _Tp&>,
      "std::expected::transform requires that F must be invocable with T");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Tp&>>;

    static_assert(__invoke_constructible<_Fun, const _Tp&>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__val_};
    } else {
      return expected<_Res, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Err2) _LIBCUDACXX_AND
             _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, _Tp2>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) && {
    static_assert(invocable<_Fun, _Tp>,
      "std::expected::transform requires that F must be invocable with T.");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Tp>>;

    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__val_));
      return expected<_Res, _Err>{};
    } else {
      return expected<_Res, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }
  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Err2) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, _Tp2>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) && {
    static_assert(invocable<_Fun, _Tp>,
      "std::expected::transform requires that F must be invocable with T");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Tp>>;

    static_assert(__invoke_constructible<_Fun, _Tp>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__val_)};
    } else {
      return expected<_Res, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, const _Err2) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, const _Tp2>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const&& {
    static_assert(invocable<_Fun, const _Tp>,
      "std::expected::transform requires that F must be invocable with T.");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Tp>>;

    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__val_));
      return expected<_Res, _Err>{};
    } else {
      return expected<_Res, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, const _Err2) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun, const _Tp2>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const&& {
    static_assert(invocable<_Fun, const _Tp>,
      "std::expected::transform requires that F must be invocable with T");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Tp>>;

    static_assert(__invoke_constructible<_Fun, const _Tp>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__val_)};
    } else {
      return expected<_Res, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Tp2, _Tp2&))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) & {
    static_assert(invocable<_Fun, _Err&>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Err&>>;

    static_assert(__invoke_constructible<_Fun, _Err&>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Tp, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Tp, _Res>{in_place, this->__union_.__val_};
    } else {
      return expected<_Tp, _Res>{
          __expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) const& {
    static_assert(invocable<_Fun, const _Err&>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Err&>>;

    static_assert(__invoke_constructible<_Fun, const _Err&>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Tp, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Tp, _Res>{in_place, this->__union_.__val_};
    } else {
      return expected<_Tp, _Res>{
          __expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) && {
    static_assert(invocable<_Fun, _Err>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Err>>;

    static_assert(__invoke_constructible<_Fun, _Err>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Tp, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Tp, _Res>{in_place, _CUDA_VSTD::move(this->__union_.__val_)};
    } else {
      return expected<_Tp, _Res>{__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun),
          _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Tp2, const _Tp2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) const&& {
    static_assert(invocable<_Fun, const _Err>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Err>>;

    static_assert(__invoke_constructible<_Fun, const _Err>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Tp, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Tp, _Res>{in_place, _CUDA_VSTD::move(this->__union_.__val_)};
    } else {
      return expected<_Tp, _Res>{__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun),
          _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  // [expected.object.eq], equality operators
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const expected& __y) {
    if (__x.__has_val_ != __y.has_value()) {
      return false;
    } else {
      if (__x.__has_val_) {
        return __x.__union_.__val_ == __y.value();
      } else {
        return __x.__union_.__unex_ == __y.error();
      }
    }
  }

#if _LIBCUDACXX_STD_VER < 20
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const expected& __x, const expected& __y) {
    return !(__x == __y);
  }
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_TEMPLATE(class _T2, class _E2)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_void, _T2)))
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const expected<_T2, _E2>& __y) {
    if (__x.__has_val_ != __y.has_value()) {
      return false;
    } else {
      if (__x.__has_val_) {
        return __x.__union_.__val_ == __y.value();
      } else {
        return __x.__union_.__unex_ == __y.error();
      }
    }
  }

#if _LIBCUDACXX_STD_VER < 20
  _LIBCUDACXX_TEMPLATE(class _T2, class _E2)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_void, _T2)))
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const expected& __x, const expected<_T2, _E2>& __y) {
    return !(__x == __y);
  }
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_TEMPLATE(class _T2)
    _LIBCUDACXX_REQUIRES( (!__expected::__is_expected_nonvoid<_T2>))
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const _T2& __v) {
    return __x.__has_val_ && static_cast<bool>(__x.__union_.__val_ == __v);
  }
#if _LIBCUDACXX_STD_VER < 20
  _LIBCUDACXX_TEMPLATE(class _T2)
    _LIBCUDACXX_REQUIRES( (!__expected::__is_expected_nonvoid<_T2>))
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const _T2& __v, const expected& __x) {
    return __x.__has_val_ && static_cast<bool>(__x.__union_.__val_ == __v);
  }
  _LIBCUDACXX_TEMPLATE(class _T2)
    _LIBCUDACXX_REQUIRES( (!__expected::__is_expected_nonvoid<_T2>))
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const expected& __x, const _T2& __v) {
    return !__x.__has_val_ || static_cast<bool>(__x.__union_.__val_ != __v);
  }
  _LIBCUDACXX_TEMPLATE(class _T2)
    _LIBCUDACXX_REQUIRES( (!__expected::__is_expected_nonvoid<_T2>))
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const _T2& __v, const expected& __x) {
    return !__x.__has_val_ || static_cast<bool>(__x.__union_.__val_ != __v);
  }
#endif // _LIBCUDACXX_STD_VER < 20

  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const unexpected<_E2>& __e) {
    return !__x.__has_val_ && static_cast<bool>(__x.__union_.__unex_ == __e.error());
  }
#if _LIBCUDACXX_STD_VER < 20
  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const unexpected<_E2>& __e, const expected& __x) {
    return !__x.__has_val_ && static_cast<bool>(__x.__union_.__unex_ == __e.error());
  }
  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const expected& __x, const unexpected<_E2>& __e) {
    return __x.__has_val_ || static_cast<bool>(__x.__union_.__unex_ != __e.error());
  }
  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const unexpected<_E2>& __e, const expected& __x) {
    return __x.__has_val_ || static_cast<bool>(__x.__union_.__unex_ != __e.error());
  }
#endif // _LIBCUDACXX_STD_VER < 20
};


template <class _Err>
class expected<void, _Err> : private __expected_move_assign<void, _Err>
                           , private __expected_void_sfinae_ctor_base_t<_Err>
                           , private __expected_void_sfinae_assign_base_t<_Err>
{
  using __base = __expected_move_assign<void, _Err>;
  static_assert(__unexpected::__valid_unexpected<_Err>,
                "[expected.void.general] A program that instantiates expected<T, E> with a E that is not a "
                "valid argument for unexpected<E> is ill-formed");

  template <class, class>
  friend class expected;

  template <class _Up, class _OtherErr, class _OtherErrQual>
  using __can_convert =
      _And< is_void<_Up>,
            is_constructible<_Err, _OtherErrQual>,
            _Not<is_constructible<unexpected<_Err>, expected<_Up, _OtherErr>&>>,
            _Not<is_constructible<unexpected<_Err>, expected<_Up, _OtherErr>>>,
            _Not<is_constructible<unexpected<_Err>, const expected<_Up, _OtherErr>&>>,
            _Not<is_constructible<unexpected<_Err>, const expected<_Up, _OtherErr>>>>;

public:
  using value_type      = void;
  using error_type      = _Err;
  using unexpected_type = unexpected<_Err>;

  template <class _Up>
  using rebind = expected<_Up, error_type>;

  // [expected.void.ctor], constructors
  constexpr expected() = default;
  constexpr expected(const expected&) = default;
  constexpr expected(expected&&) = default;
  constexpr expected& operator=(const expected&) = default;
  constexpr expected& operator=(expected&&) = default;

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, const _OtherErr&>::value _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected(const expected<_Up, _OtherErr>& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(__other.__has_val_)
  {
    if (!__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, const _OtherErr&>::value _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  explicit expected(const expected<_Up, _OtherErr>& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(__other.__has_val_)
  {
    if (!__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, _OtherErr>::value _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected(expected<_Up, _OtherErr>&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(__other.__has_val_)
  {
    if (!__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up, class _OtherErr)
    _LIBCUDACXX_REQUIRES( __can_convert<_Up, _OtherErr, _OtherErr>::value _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  explicit expected(expected<_Up, _OtherErr>&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(__other.__has_val_)
  {
    if (!__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, const _OtherErr&) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(const unexpected<_OtherErr>& __unex)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(unexpect, __unex.error())
  {}

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, const _OtherErr&) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_convertible, const _OtherErr&, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(const unexpected<_OtherErr>& __unex)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
    : __base(unexpect, __unex.error())
  {}

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _OtherErr) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(unexpected<_OtherErr>&& __unex)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(unexpect, _CUDA_VSTD::move(__unex.error()))
  {}

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _OtherErr) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_convertible, _OtherErr, _Err))
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(unexpected<_OtherErr>&& __unex)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr)) // strengthened
    : __base(unexpect, _CUDA_VSTD::move(__unex.error()))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(in_place_t) noexcept : __base(true) {}

  _LIBCUDACXX_TEMPLATE(class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...)) // strengthened
    : __base(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _LIBCUDACXX_TEMPLATE(class _Up, class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, initializer_list<_Up>&, _Args...))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit expected(unexpect_t, initializer_list<_Up> __il, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, initializer_list<_Up>, _Args...)) // strengthened
    : __base(unexpect, __il, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

private:
  template<class _Fun, class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  expected(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __base(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
  {}

public:

  // [expected.void.dtor], destructor
  // [expected.void.assign], assignment
  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, const _OtherErr&) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_assignable, _Err&, const _OtherErr&)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected& operator=(const unexpected<_OtherErr>& __un)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_assignable, _Err&, const _OtherErr&) &&
             _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, const _OtherErr&)) // strengthened
  {
    if (this->__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __un.error());
      this->__has_val_ = false;
    } else {
      this->__union_.__unex_ = __un.error();
    }
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherErr)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err, _OtherErr) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_assignable, _Err&, _OtherErr)
    )
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  expected& operator=(unexpected<_OtherErr>&& __un)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_assignable, _Err&, _OtherErr) &&
             _LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _OtherErr))
  {
    if (this->__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__un.error()));
      this->__has_val_ = false;
    } else {
      this->__union_.__unex_ = _CUDA_VSTD::move(__un.error());
    }
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  void emplace() noexcept
  {
    if (!this->__has_val_) {
      this->__union_.__unex_.~_Err();
      this->__has_val_ = true;
    }
  }

  // [expected.void.swap], swap
  _LIBCUDACXX_TEMPLATE(class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( __expected::__can_swap<void, _Err2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void swap(expected<void, _Err2>& __rhs) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err2) &&
                                                   _LIBCUDACXX_TRAIT(is_nothrow_swappable, _Err2))
  {
    if (this->__has_val_) {
      if (!__rhs.__has_val_) {
        this->__swap_val_unex_impl(*this, __rhs);
      }
    } else {
      if (__rhs.__has_val_) {
        this->__swap_val_unex_impl(__rhs, *this);
      } else {
        using _CUDA_VSTD::swap;
        swap(this->__union_.__unex_, __rhs.__union_.__unex_);
      }
    }
  }

  template<class _Err2 = _Err>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  auto swap(expected& __x, expected& __y) noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err2) &&
                                                   _LIBCUDACXX_TRAIT(is_nothrow_swappable, _Err2))
    _LIBCUDACXX_TRAILING_REQUIRES(void)(__expected::__can_swap<void, _Err2>)
  {
    return __x.swap(__y); // some compiler warn about non void function without return
  }

  // [expected.void.obs], observers
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr explicit operator bool() const noexcept { return this->__has_val_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr bool has_value() const noexcept { return this->__has_val_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr void operator*() const noexcept {
    _LIBCUDACXX_ASSERT(this->__has_val_, "expected::operator* requires the expected to contain a value");
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr void value() const& {
    if (!this->__has_val_) {
      __expected::__throw_bad_expected_access<_Err>(this->__union_.__unex_);
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr void value() && {
    if (!this->__has_val_) {
      __expected::__throw_bad_expected_access<_Err>(_CUDA_VSTD::move(this->__union_.__unex_));
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr const _Err& error() const& noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return this->__union_.__unex_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Err& error() & noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return this->__union_.__unex_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr const _Err&& error() const&& noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return _CUDA_VSTD::move(this->__union_.__unex_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Err&& error() && noexcept {
    _LIBCUDACXX_ASSERT(!this->__has_val_, "expected::error requires the expected to contain an error");
    return _CUDA_VSTD::move(this->__union_.__unex_);
  }

  // [expected.void.monadic]
  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, _Err2&))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) & {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
    } else {
      return _Res{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Err2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) const& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
    } else {
      return _Res{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Err2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) && {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
    } else {
      return _Res{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, const _Err2))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto and_then(_Fun&& __fun) const&& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of f(value()) must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::error_type, _Err),
      "The error type of the result of f(value()) must be the same as that of std::expected");

    if (this->__has_val_) {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
    } else {
      return _Res{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) & {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, _Err&>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, void),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_);
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) const& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, const _Err&>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, void),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_);
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) && {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, _Err>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, void),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__unex_));
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto or_else(_Fun&& __fun) const&& {
    using _Res = __remove_cvref_t<invoke_result_t<_Fun, const _Err>>;

    static_assert(__expected::__is_expected<_Res>,
      "Result of std::expected::or_else must be a specialization of std::expected");
    static_assert(_LIBCUDACXX_TRAIT(is_same, typename _Res::value_type, void),
      "The value type of the result of std::expected::or_else must be the same as that of std::expected");

    if (this->__has_val_) {
      return _Res{};
    } else {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::move(this->__union_.__unex_));
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, _Err2&) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) & {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T.");
    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
      return expected<void, _Err>{};
    } else {
      return expected<void, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, _Err2&) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) & {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T.");
    using _Res = __remove_cv_t<invoke_result_t<_Fun>>;

    static_assert(__invoke_constructible<_Fun>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun)};
    } else {
      return expected<_Res, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Err2) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const& {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T.");
    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
      return expected<void, _Err>{};
    } else {
      return expected<void, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_copy_constructible, _Err2) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const& {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T");
    using _Res = __remove_cv_t<invoke_result_t<_Fun>>;

    static_assert(__invoke_constructible<_Fun>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun)};
    } else {
      return expected<_Res, _Err>{unexpect, this->__union_.__unex_};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Err2) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) && {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T.");
    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
      return expected<void, _Err>{};
    } else {
      return expected<void, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }
  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_move_constructible, _Err2) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) && {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T");
    using _Res = __remove_cv_t<invoke_result_t<_Fun>>;

    static_assert(__invoke_constructible<_Fun>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun)};
    } else {
      return expected<_Res, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, const _Err2) _LIBCUDACXX_AND
              _LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const&& {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T.");
    if (this->__has_val_) {
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun));
      return expected<void, _Err>{};
    } else {
        return expected<void, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Fun, class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_constructible, _Err2, const _Err2) _LIBCUDACXX_AND
            (!_LIBCUDACXX_TRAIT(is_same, __remove_cv_t<invoke_result_t<_Fun>>, void)))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform(_Fun&& __fun) const&& {
    static_assert(invocable<_Fun>,
      "std::expected::transform requires that F must be invocable with T");
    using _Res = __remove_cv_t<invoke_result_t<_Fun>>;

    static_assert(__invoke_constructible<_Fun>,
      "std::expected::transform requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<_Res, _Err>,
      "std::expected::transform requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<_Res, _Err>{
              __expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun)};
    } else {
        return expected<_Res, _Err>{unexpect, _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) & {
    static_assert(invocable<_Fun, _Err&>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Err&>>;

    static_assert(__invoke_constructible<_Fun, _Err&>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<void, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<void, _Res>{};
    } else {
      return expected<void, _Res>{
          __expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_};
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) const& {
    static_assert(invocable<_Fun, const _Err&>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Err&>>;

    static_assert(__invoke_constructible<_Fun, const _Err&>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<void, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<void, _Res>{};
    } else {
      return expected<void, _Res>{
          __expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), this->__union_.__unex_};
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) && {
    static_assert(invocable<_Fun, _Err>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, _Err>>;

    static_assert(__invoke_constructible<_Fun, _Err>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<void, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<void, _Res>{};
    } else {
      return expected<void, _Res>{__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun),
          _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  template<class _Fun>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  auto transform_error(_Fun&& __fun) const&& {
    static_assert(invocable<_Fun, const _Err>,
      "std::expected::transform_error requires that F must be invocable with E");
    using _Res = __remove_cv_t<invoke_result_t<_Fun, const _Err>>;

    static_assert(__invoke_constructible<_Fun, const _Err>,
      "std::expected::transform_error requires that the return type of F is constructible with the result of invoking F");
    static_assert(__expected::__valid_expected<void, _Res>,
      "std::expected::transform_error requires that the return type of F must be a valid argument for std::expected");

    if (this->__has_val_) {
      return expected<void, _Res>{};
    } else {
      return expected<void, _Res>{__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun),
          _CUDA_VSTD::move(this->__union_.__unex_)};
    }
  }

  // [expected.void.eq], equality operators
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const expected& __y) noexcept {
    if (__x.__has_val_ != __y.has_value()) {
      return false;
    } else {
      return __x.__has_val_ || static_cast<bool>(__x.__union_.__unex_ == __y.error());
    }
  }
#if _LIBCUDACXX_STD_VER < 20
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const expected& __x, const expected& __y) noexcept {
    return !(__x == __y);
  }
#endif

  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const expected<void, _E2>& __y) noexcept {
    if (__x.__has_val_ != __y.has_value()) {
      return false;
    } else {
      return __x.__has_val_ || static_cast<bool>(__x.__union_.__unex_ == __y.error());
    }
  }
#if _LIBCUDACXX_STD_VER < 20
  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator!=(const expected& __x, const expected<void, _E2>& __y) noexcept {
    return !(__x == __y);
  }
#endif

  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const expected& __x, const unexpected<_E2>& __y) noexcept {
    return !__x.__has_val_ && static_cast<bool>(__x.__union_.__unex_ == __y.error());
  }
#if _LIBCUDACXX_STD_VER < 20
  template <class _E2>
  friend _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool operator==(const unexpected<_E2>& __y, const expected& __x) noexcept {
    return !__x.__has_val_ && static_cast<bool>(__x.__union_.__unex_ == __y.error());
  }
  template <class _E2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(const expected& __x, const unexpected<_E2>& __y) noexcept {
    return __x.__has_val_ || static_cast<bool>(__x.__union_.__unex_ != __y.error());
  }
  template <class _E2>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr bool operator!=(const unexpected<_E2>& __y, const expected& __x) noexcept {
    return __x.__has_val_ || static_cast<bool>(__x.__union_.__unex_ != __y.error());
  }
#endif
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___EXPECTED_EXPECTED_H
