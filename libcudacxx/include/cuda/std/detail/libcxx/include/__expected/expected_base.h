//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___EXPECTED_EXPECTED_BASE_H
#define _LIBCUDACXX___EXPECTED_EXPECTED_BASE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__concepts/__concept_macros.h"
#include "../__concepts/invocable.h"
#include "../__expected/unexpect.h"
#include "../__memory/addressof.h"
#include "../__memory/construct_at.h"
#include "../__tuple_dir/sfinae_helpers.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_copy_assignable.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__type_traits/is_default_constructible.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_assignable.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_trivially_copy_assignable.h"
#include "../__type_traits/is_trivially_copy_constructible.h"
#include "../__type_traits/is_trivially_destructible.h"
#include "../__type_traits/is_trivially_move_assignable.h"
#include "../__type_traits/is_trivially_move_constructible.h"
#include "../__type_traits/is_void.h"
#include "../__utility/exception_guard.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// MSVC complains about [[no_unique_address]] prior to C++20 as a vendor extension
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable : 4848)
#endif // _LIBCUDACXX_COMPILER_MSVC

struct __expected_construct_from_invoke_tag {
  explicit __expected_construct_from_invoke_tag() = default;
};

template <class _Tp, class _Err,
          bool = _LIBCUDACXX_TRAIT(is_trivially_destructible, _Tp)
              && _LIBCUDACXX_TRAIT(is_trivially_destructible, _Err)>
union __expected_union_t {
  struct __empty_t {};

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_default_constructible, _Tp2))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t() noexcept(_LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _Tp2)) : __val_() {}

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_default_constructible, _Tp2)))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t() noexcept : __empty_() {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(in_place_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __val_(_CUDA_VSTD::forward<_Args>(__args)...) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
    : __unex_(_CUDA_VSTD::forward<_Args>(__args)...) {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __val_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)) {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)) {}

  // the __expected_destruct's destructor handles this
  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__expected_union_t() {}

  _LIBCUDACXX_NO_UNIQUE_ADDRESS __empty_t __empty_;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Err __unex_;
};

template <class _Tp, class _Err>
union __expected_union_t<_Tp, _Err, true> {
  struct __empty_t {};

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_default_constructible, _Tp2))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t() noexcept(_LIBCUDACXX_TRAIT(is_nothrow_default_constructible, _Tp2)) : __val_() {}

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_default_constructible, _Tp2)))
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t() noexcept : __empty_() {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(in_place_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __val_(_CUDA_VSTD::forward<_Args>(__args)...) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
    : __unex_(_CUDA_VSTD::forward<_Args>(__args)...) {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __val_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)) {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_union_t(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)) {}

  _LIBCUDACXX_NO_UNIQUE_ADDRESS __empty_t __empty_;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __val_;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Err __unex_;
};

template <class _Tp, class _Err,
          bool = _LIBCUDACXX_TRAIT(is_trivially_destructible, _Tp),
          bool = _LIBCUDACXX_TRAIT(is_trivially_destructible, _Err)>
struct __expected_destruct;

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, false, false> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  constexpr __expected_destruct() noexcept = default;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(const bool __has_val) noexcept : __has_val_(__has_val) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(in_place_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
    : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__expected_destruct() {
    if (__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__union_.__val_));
    } else {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__union_.__unex_));
    }
  }
};

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, true, false> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  constexpr __expected_destruct() noexcept = default;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(const bool __has_val) noexcept : __has_val_(__has_val) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(in_place_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__expected_destruct() {
    if (!__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__union_.__unex_));
    }
  }
};

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, false, true> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  constexpr __expected_destruct() noexcept = default;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(const bool __has_val) noexcept : __has_val_(__has_val) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(in_place_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__expected_destruct() {
    if (__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__union_.__val_));
    }
  }
};

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, true, true> {
  // This leads to an ICE with nvcc, see nvbug4103076
  /* _LIBCUDACXX_NO_UNIQUE_ADDRESS */ __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  constexpr __expected_destruct() noexcept = default;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(const bool __has_val) noexcept : __has_val_(__has_val) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(in_place_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, _Args...))
    : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, in_place_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, in_place, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(true)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}
};

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(pop)
#endif // _LIBCUDACXX_COMPILER_MSVC

template <class _Tp, class _Err>
struct __expected_storage : __expected_destruct<_Tp, _Err>
{
  using __base = __expected_destruct<_Tp, _Err>;

// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __expected_storage() noexcept = default;

  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_storage(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  _LIBCUDACXX_TEMPLATE(class _T1, class _T2, class... _Args)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _Args...))
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __reinit_expected(_T1& __newval, _T2& __oldval, _Args&&... __args) noexcept {
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__oldval));
    _LIBCUDACXX_CONSTRUCT_AT(__newval, _CUDA_VSTD::forward<_Args>(__args)...);
  }

  _LIBCUDACXX_TEMPLATE(class _T1, class _T2, class... _Args)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _Args...)) _LIBCUDACXX_AND
                _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _T1)
    )
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __reinit_expected(_T1& __newval, _T2& __oldval, _Args&&... __args) {
    _T1 __tmp(_CUDA_VSTD::forward<_Args>(__args)...);
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__oldval));
    _LIBCUDACXX_CONSTRUCT_AT(__newval, _CUDA_VSTD::move(__tmp));
  }

  _LIBCUDACXX_TEMPLATE(class _T1, class _T2, class... _Args)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_nothrow_constructible, _T1, _Args...)) _LIBCUDACXX_AND
              (!_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _T1))
    )
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __reinit_expected(_T1& __newval, _T2& __oldval, _Args&&... __args) {
    static_assert(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _T2),
        "To provide strong exception guarantee, T2 has to satisfy `is_nothrow_move_constructible_v` so that it can "
        "be reverted to the previous state in case an exception is thrown during the assignment.");
    _T2 __tmp(_CUDA_VSTD::move(__oldval));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__oldval));
    auto __trans =
        _CUDA_VSTD::__make_exception_guard([&] { _LIBCUDACXX_CONSTRUCT_AT(__oldval, _CUDA_VSTD::move(__tmp)); });
    _LIBCUDACXX_CONSTRUCT_AT(__newval, _CUDA_VSTD::forward<_Args>(__args)...);
    __trans.__complete();
  }

  _LIBCUDACXX_TEMPLATE(class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err2))
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __swap_val_unex_impl(__expected_storage<_Tp, _Err2>& __with_val, __expected_storage& __with_err) {
    _Err __tmp(_CUDA_VSTD::move(__with_err.__union_.__unex_));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_));
    auto __trans = _CUDA_VSTD::__make_exception_guard([&] {
      _LIBCUDACXX_CONSTRUCT_AT(__with_err.__union_.__unex_, _CUDA_VSTD::move(__tmp));
    });
    _LIBCUDACXX_CONSTRUCT_AT(__with_err.__union_.__val_, _CUDA_VSTD::move(__with_val.__union_.__val_));
    __trans.__complete();
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_val.__union_.__val_));
    _LIBCUDACXX_CONSTRUCT_AT(__with_val.__union_.__unex_, _CUDA_VSTD::move(__tmp));
    __with_val.__has_val_ = false;
    __with_err.__has_val_ = true;
  }

  _LIBCUDACXX_TEMPLATE(class _Err2 = _Err)
    _LIBCUDACXX_REQUIRES( (!_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err2)))
  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __swap_val_unex_impl(__expected_storage<_Tp, _Err2>& __with_val, __expected_storage& __with_err) {
    static_assert(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp),
                  "To provide strong exception guarantee, Tp has to satisfy `is_nothrow_move_constructible_v` so "
                  "that it can be reverted to the previous state in case an exception is thrown during swap.");
    _Tp __tmp(_CUDA_VSTD::move(__with_val.__union_.__val_));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_val.__union_.__val_));
    auto __trans = _CUDA_VSTD::__make_exception_guard([&] {
      _LIBCUDACXX_CONSTRUCT_AT(__with_val.__union_.__val_, _CUDA_VSTD::move(__tmp));
    });
    _LIBCUDACXX_CONSTRUCT_AT(__with_val.__union_.__unex_, _CUDA_VSTD::move(__with_err.__union_.__unex_));
    __trans.__complete();
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_));
    _LIBCUDACXX_CONSTRUCT_AT(__with_err.__union_.__val_, _CUDA_VSTD::move(__tmp));
    __with_val.__has_val_ = false;
    __with_err.__has_val_ = true;
  }
};

template <class _Tp, class _Err, bool =
          (_LIBCUDACXX_TRAIT(is_trivially_copy_constructible, _Tp) || _LIBCUDACXX_TRAIT(is_same, _Tp, void)) &&
           _LIBCUDACXX_TRAIT(is_trivially_copy_constructible, _Err)>
struct __expected_copy : __expected_storage<_Tp, _Err>
{
  using __base = __expected_storage<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __expected_copy() noexcept = default;

  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_copy(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp, class _Err>
struct __expected_copy<_Tp, _Err, false> : __expected_storage<_Tp, _Err>
{
  using __base = __expected_storage<_Tp, _Err>;

// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_copy(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __expected_copy() noexcept = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_copy(const __expected_copy& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _Tp)
          && _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _Err))
    : __base(__other.__has_val_)
  {
    if (__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, __other.__union_.__val_);
    } else {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
    }
  }

  __expected_copy(__expected_copy&&) = default;
  __expected_copy& operator=(const __expected_copy&) = default;
  __expected_copy& operator=(__expected_copy&&) = default;
};

template <class _Tp, class _Err, bool =
          (_LIBCUDACXX_TRAIT(is_trivially_move_constructible, _Tp) || _LIBCUDACXX_TRAIT(is_same, _Tp, void)) &&
           _LIBCUDACXX_TRAIT(is_trivially_move_constructible, _Err)>
struct __expected_move : __expected_copy<_Tp, _Err>
{
  using __base = __expected_copy<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __expected_move() noexcept = default;

  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_move(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp, class _Err>
struct __expected_move<_Tp, _Err, false> : __expected_copy<_Tp, _Err>
{
  using __base = __expected_copy<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_move(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  __expected_move() = default;
  __expected_move(const __expected_move&) = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_move(__expected_move&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp)
          && _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err))
    : __base(__other.__has_val_)
  {
    if (__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__val_, _CUDA_VSTD::move(__other.__union_.__val_));
    } else {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  __expected_move& operator=(const __expected_move&) = default;
  __expected_move& operator=(__expected_move&&) = default;
};

template <class _Tp, class _Err, bool =
          (_LIBCUDACXX_TRAIT(is_trivially_destructible, _Tp)  || _LIBCUDACXX_TRAIT(is_same, _Tp, void)) &&
           _LIBCUDACXX_TRAIT(is_trivially_destructible, _Err) &&
          (_LIBCUDACXX_TRAIT(is_trivially_copy_constructible, _Tp)  || _LIBCUDACXX_TRAIT(is_same, _Tp, void))&&
           _LIBCUDACXX_TRAIT(is_trivially_copy_constructible, _Err) &&
          (_LIBCUDACXX_TRAIT(is_trivially_copy_assignable, _Tp)  || _LIBCUDACXX_TRAIT(is_same, _Tp, void))&&
           _LIBCUDACXX_TRAIT(is_trivially_copy_assignable, _Err)>
struct __expected_copy_assign : __expected_move<_Tp, _Err>
{
  using __base = __expected_move<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __expected_copy_assign() noexcept = default;

  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_copy_assign(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp, class _Err>
struct __expected_copy_assign<_Tp, _Err, false> : __expected_move<_Tp, _Err>
{
  using __base = __expected_move<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_copy_assign(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  __expected_copy_assign() = default;
  __expected_copy_assign(const __expected_copy_assign&) = default;
  __expected_copy_assign(__expected_copy_assign&&) = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_copy_assign& operator=(const __expected_copy_assign& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _Tp) &&
             _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _Tp) &&
             _LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _Err) &&
             _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_) {
      this->__union_.__val_ = __other.__union_.__val_;
    } else if (this->__has_val_ && !__other.__has_val_) {
      this->__reinit_expected(this->__union_.__unex_, this->__union_.__val_, __other.__union_.__unex_);
      this->__has_val_ = false;
    } else if (!this->__has_val_ && __other.__has_val_) {
      this->__reinit_expected(this->__union_.__val_, this->__union_.__unex_, __other.__union_.__val_);
      this->__has_val_ = true;
    } else { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = __other.__union_.__unex_;
    }
    return *this;
  }

  __expected_copy_assign& operator=(__expected_copy_assign&&) = default;
};

template <class _Tp, class _Err, bool =
          (_LIBCUDACXX_TRAIT(is_trivially_destructible, _Tp) || _LIBCUDACXX_TRAIT(is_same, _Tp, void)) &&
          _LIBCUDACXX_TRAIT(is_trivially_destructible, _Err) &&
          (_LIBCUDACXX_TRAIT(is_trivially_move_constructible, _Tp) || _LIBCUDACXX_TRAIT(is_same, _Tp, void)) &&
          _LIBCUDACXX_TRAIT(is_trivially_move_constructible, _Err) &&
          (_LIBCUDACXX_TRAIT(is_trivially_move_assignable, _Tp) || _LIBCUDACXX_TRAIT(is_same, _Tp, void)) &&
          _LIBCUDACXX_TRAIT(is_trivially_move_assignable, _Err)>
struct __expected_move_assign : __expected_copy_assign<_Tp, _Err>
{
  using __base = __expected_copy_assign<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __expected_move_assign() noexcept = default;

  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_move_assign(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3
};

template <class _Tp, class _Err>
struct __expected_move_assign<_Tp, _Err, false> : __expected_copy_assign<_Tp, _Err>
{
  using __base = __expected_copy_assign<_Tp, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_move_assign(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  __expected_move_assign() = default;
  __expected_move_assign(const __expected_move_assign&) = default;
  __expected_move_assign(__expected_move_assign&&) = default;
  __expected_move_assign& operator=(const __expected_move_assign&) = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_move_assign& operator=(__expected_move_assign&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_assignable, _Tp) &&
             _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp) &&
             _LIBCUDACXX_TRAIT(is_nothrow_move_assignable, _Err) &&
             _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_) {
      this->__union_.__val_ = _CUDA_VSTD::move(__other.__union_.__val_);
    } else if (this->__has_val_ && !__other.__has_val_) {
      this->__reinit_expected(this->__union_.__unex_, this->__union_.__val_, _CUDA_VSTD::move(__other.__union_.__unex_));
      this->__has_val_ = false;
    } else if (!this->__has_val_ && __other.__has_val_) {
      this->__reinit_expected(this->__union_.__val_, this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__val_));
      this->__has_val_ = true;
    } else { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = _CUDA_VSTD::move(__other.__union_.__unex_);
    }
    return *this;
  }
};

template <class _Tp, class _Err>
using __expected_sfinae_ctor_base_t = __sfinae_ctor_base<
  _LIBCUDACXX_TRAIT(is_copy_constructible, _Tp) && _LIBCUDACXX_TRAIT(is_copy_constructible, _Err),
  _LIBCUDACXX_TRAIT(is_move_constructible, _Tp) && _LIBCUDACXX_TRAIT(is_move_constructible, _Err)
>;

template <class _Tp, class _Err>
using __expected_sfinae_assign_base_t = __sfinae_assign_base<
  _LIBCUDACXX_TRAIT(is_copy_constructible, _Tp) && _LIBCUDACXX_TRAIT(is_copy_constructible, _Err) &&
  _LIBCUDACXX_TRAIT(is_copy_assignable, _Tp)    && _LIBCUDACXX_TRAIT(is_copy_assignable, _Err) &&
  (_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp) || _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err)),
  _LIBCUDACXX_TRAIT(is_move_constructible, _Tp) && _LIBCUDACXX_TRAIT(is_move_constructible, _Err) &&
  _LIBCUDACXX_TRAIT(is_move_assignable, _Tp)    && _LIBCUDACXX_TRAIT(is_move_assignable, _Err) &&
 (_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Tp) || _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err))
>;

// expected<void, E> base classtemplate <class _Tp, class _Err>
// MSVC complains about [[no_unique_address]] prior to C++20 as a vendor extension
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable : 4848)
#endif // _LIBCUDACXX_COMPILER_MSVC

template <class _Err>
struct __expected_destruct<void, _Err, false, false> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS union __expected_union_t {
    struct __empty_t {};

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __expected_union_t() noexcept : __empty_() {}

    template<class... _Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __expected_union_t(unexpect_t, _Args&&... __args)
      noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __unex_(_CUDA_VSTD::forward<_Args>(__args)...) {}

    template<class _Fun, class... _Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __expected_union_t(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
      noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)) {}

    // the __expected_destruct's destructor handles this
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    ~__expected_union_t() {}

    _LIBCUDACXX_NO_UNIQUE_ADDRESS __empty_t __empty_;
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Err __unex_;
  } __union_{};
  bool __has_val_{true};

  constexpr __expected_destruct() noexcept = default;

  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(const bool __has_val) noexcept : __has_val_(__has_val) {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
    : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__expected_destruct() {
    if (!__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__union_.__unex_));
    }
  }
};

template <class _Err>
struct __expected_destruct<void, _Err, false, true> {
  // Using `_LIBCUDACXX_NO_UNIQUE_ADDRESS` here crashes nvcc
  /* _LIBCUDACXX_NO_UNIQUE_ADDRESS */ union __expected_union_t {
    struct __empty_t {};

    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __expected_union_t() noexcept : __empty_() {}

    template<class... _Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __expected_union_t(unexpect_t, _Args&&... __args)
      noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __unex_(_CUDA_VSTD::forward<_Args>(__args)...) {}

    template<class _Fun, class... _Args>
    _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __expected_union_t(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
      noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)) {}

    _LIBCUDACXX_NO_UNIQUE_ADDRESS __empty_t __empty_;
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Err __unex_;
  } __union_{};
  bool __has_val_{true};

  constexpr __expected_destruct() noexcept = default;

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(in_place_t)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
    : __union_()
    , __has_val_(true)
  {}

  template<class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(unexpect_t, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, _Args...))
    : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  template<class _Fun, class... _Args>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(__expected_construct_from_invoke_tag, unexpect_t, _Fun&& __fun, _Args&&... __args)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
    : __union_(__expected_construct_from_invoke_tag{}, unexpect, _CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...)
    , __has_val_(false)
  {}

  _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_destruct(const bool __has_val) noexcept : __has_val_(__has_val) {}
};

#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(pop)
#endif // _LIBCUDACXX_COMPILER_MSVC

template <class _Err>
struct __expected_storage<void, _Err> : __expected_destruct<void, _Err>
{
  using __base = __expected_destruct<void, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  constexpr __expected_storage() noexcept = default;

  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_storage(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  static _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __swap_val_unex_impl(__expected_storage& __with_val, __expected_storage& __with_err)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err)) {
    _LIBCUDACXX_CONSTRUCT_AT(__with_val.__union_.__unex_, _CUDA_VSTD::move(__with_err.__union_.__unex_));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_));
    __with_val.__has_val_ = false;
    __with_err.__has_val_ = true;
  }
};

template <class _Err>
struct __expected_copy<void, _Err, false> : __expected_storage<void, _Err>
{
  using __base = __expected_storage<void, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_copy(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  constexpr __expected_copy() = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_copy(const __expected_copy& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _Err))
    : __base(__other.__has_val_)
  {
    if (!__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
    }
  }

  __expected_copy(__expected_copy&&) = default;
  __expected_copy& operator=(const __expected_copy&) = default;
  __expected_copy& operator=(__expected_copy&&) = default;
};

template <class _Err>
struct __expected_move<void, _Err, false> : __expected_copy<void, _Err>
{
  using __base = __expected_copy<void, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_move(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  __expected_move() = default;
  __expected_move(const __expected_move&) = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_move(__expected_move&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err))
    : __base(__other.__has_val_)
  {
    if (!__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  __expected_move& operator=(const __expected_move&) = default;
  __expected_move& operator=(__expected_move&&) = default;
};

template <class _Err>
struct __expected_copy_assign<void, _Err, false> : __expected_move<void, _Err>
{
  using __base = __expected_move<void, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_copy_assign(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  __expected_copy_assign() = default;
  __expected_copy_assign(const __expected_copy_assign&) = default;
  __expected_copy_assign(__expected_copy_assign&&) = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_copy_assign& operator=(const __expected_copy_assign& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_copy_assignable, _Err) &&
             _LIBCUDACXX_TRAIT(is_nothrow_copy_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_) {
      // nothing to do
    } else if (this->__has_val_ && !__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, __other.__union_.__unex_);
      this->__has_val_ = false;
    } else if (!this->__has_val_ && __other.__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__unex_));
      this->__has_val_ = true;
    } else { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = __other.__union_.__unex_;
    }
    return *this;
  }

  __expected_copy_assign& operator=(__expected_copy_assign&&) = default;
};

template <class _Err>
struct __expected_move_assign<void, _Err, false> : __expected_copy_assign<void, _Err>
{
  using __base = __expected_copy_assign<void, _Err>;
// nvbug3961621
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_CUDACC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
  template<class... _Args, __enable_if_t<_LIBCUDACXX_TRAIT(is_constructible, __base, _Args...), int> = 0>
   _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __expected_move_assign(_Args&&... __args) noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
    : __base(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

  __expected_move_assign() = default;
  __expected_move_assign(const __expected_move_assign&) = default;
  __expected_move_assign(__expected_move_assign&&) = default;
  __expected_move_assign& operator=(const __expected_move_assign&) = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __expected_move_assign& operator=(__expected_move_assign&& __other)
    noexcept(_LIBCUDACXX_TRAIT(is_nothrow_move_assignable, _Err) &&
             _LIBCUDACXX_TRAIT(is_nothrow_move_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_) {
      // nothing to do
    } else if (this->__has_val_ && !__other.__has_val_) {
      _LIBCUDACXX_CONSTRUCT_AT(this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__unex_));
      this->__has_val_ = false;
    } else if (!this->__has_val_ && __other.__has_val_) {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__unex_));
      this->__has_val_ = true;
    } else { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = _CUDA_VSTD::move(__other.__union_.__unex_);
    }
    return *this;
  }
};

template <class _Err>
using __expected_void_sfinae_ctor_base_t = __sfinae_ctor_base<
  _LIBCUDACXX_TRAIT(is_copy_constructible, _Err),
  _LIBCUDACXX_TRAIT(is_move_constructible, _Err)
>;

template <class _Err>
using __expected_void_sfinae_assign_base_t = __sfinae_assign_base<
  _LIBCUDACXX_TRAIT(is_copy_constructible, _Err) &&
  _LIBCUDACXX_TRAIT(is_copy_assignable, _Err),
  _LIBCUDACXX_TRAIT(is_move_constructible, _Err) &&
  _LIBCUDACXX_TRAIT(is_move_assignable, _Err)
>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_STD_VER > 11

#endif // _LIBCUDACXX___EXPECTED_EXPECTED_BASE_H
