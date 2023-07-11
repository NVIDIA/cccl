// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_VARIANT_LIKE_H
#define _LIBCUDACXX___RANGES_VARIANT_LIKE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/swappable.h"
#include "../__memory/addressof.h"
#include "../__memory/construct_at.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_assignable.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_swappable.h"
#include "../__type_traits/is_trivially_copy_assignable.h"
#include "../__type_traits/is_trivially_copy_constructible.h"
#include "../__type_traits/is_trivially_destructible.h"
#include "../__type_traits/is_trivially_move_assignable.h"
#include "../__type_traits/is_trivially_move_constructible.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(push)
#pragma warning(disable : 4848)
#endif // _LIBCUDACXX_COMPILER_MSVC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

template <class _Tp>
_LIBCUDACXX_CONCEPT __is_trivially_swappable_v = is_trivially_destructible_v<_Tp>&&
    is_trivially_move_constructible_v<_Tp> && is_trivially_move_assignable_v<_Tp> &&
    !_CUDA_VRANGES::__swap::__unqualified_swappable_with<_Tp, _Tp>;

// __variant_like is a simplified varaint with just two alternatives that does not need all the complexity of variant

enum class __variant_like_state : unsigned char { __nothing, __holds_first, __holds_second };

struct __construct_first{};
struct __construct_second{};

template <class _Tp, class _Up, bool = is_trivially_destructible_v<_Tp> &&
                                       is_trivially_destructible_v<_Up>>
struct __vl_destruct_base {
  union {
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __first_;
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Up __second_;
  };
  __variant_like_state __contains_;

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    (requires default_initializable<_Tp2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __vl_destruct_base() noexcept(is_nothrow_default_constructible_v<_Tp2>)
    : __first_(), __contains_{__variant_like_state::__holds_first} {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __vl_destruct_base(__variant_like_state __state) noexcept : __contains_{__state} {}

  template <class... _Types>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __vl_destruct_base(__construct_first, _Types&&... _Args) noexcept(
    is_nothrow_constructible_v<_Tp, _Types...>)
    : __first_(_CUDA_VSTD::forward<_Types>(_Args)...), __contains_{__variant_like_state::__holds_first}
  {}

  _LIBCUDACXX_TEMPLATE(class... _Types)
    (requires (!same_as<_Tp, _Up>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __vl_destruct_base(__construct_second, _Types&&... _Args) noexcept(
    is_nothrow_constructible_v<_Up, _Types...>)
    : __second_(_CUDA_VSTD::forward<_Types>(_Args)...), __contains_{__variant_like_state::__holds_second}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  ~__vl_destruct_base() {
    __raw_clear();
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __raw_clear() noexcept {
    switch (__contains_) {
    case __variant_like_state::__holds_first:
        __first_.~_Tp();
        break;
    case __variant_like_state::__holds_second:
        __second_.~_Up();
        break;
    case __variant_like_state::__nothing:
        break;
    }
  }
};

template <class _Tp, class _Up>
struct __vl_destruct_base<_Tp, _Up, true> {
  union {
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Tp __first_;
    _LIBCUDACXX_NO_UNIQUE_ADDRESS _Up __second_;
  };
  __variant_like_state __contains_;

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
    (requires default_initializable<_Tp2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  __vl_destruct_base() noexcept(is_nothrow_default_constructible_v<_Tp2>)
    : __first_(), __contains_{__variant_like_state::__holds_first} {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __vl_destruct_base(__variant_like_state __state) noexcept : __contains_{__state} {}

  template <class... _Types>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __vl_destruct_base(__construct_first, _Types&&... _Args) noexcept(
    is_nothrow_constructible_v<_Tp, _Types...>)
    : __first_(_CUDA_VSTD::forward<_Types>(_Args)...), __contains_{__variant_like_state::__holds_first}
  {}

  _LIBCUDACXX_TEMPLATE(class... _Types)
    (requires (!same_as<_Tp, _Up>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  explicit __vl_destruct_base(__construct_second, _Types&&... _Args) noexcept(
    is_nothrow_constructible_v<_Up, _Types...>)
    : __second_(_CUDA_VSTD::forward<_Types>(_Args)...), __contains_{__variant_like_state::__holds_second}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr void __raw_clear() noexcept {}
};

template <class _Tp, class _Up>
class __variant_like : public __vl_destruct_base<_Tp, _Up> {
public:
  using __base = __vl_destruct_base<_Tp, _Up>;
#if defined(_LIBCUDACXX_COMPILER_NVRTC)  \
 || (defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3) && defined(_LIBCUDACXX_COMPILER_CLANG))
    template<class... _Args>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
    __variant_like(_Args&&... __args)
      noexcept(noexcept(__base(_CUDA_VSTD::declval<_Args>()...)))
      : __base(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
#else // ^^^ _LIBCUDACXX_COMPILER_NVRTC || nvcc < 11.3 ^^^ / vvv !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3 vvv
  using __base::__base;
#endif // !_LIBCUDACXX_COMPILER_NVRTC || nvcc >= 11.3

    constexpr __variant_like() noexcept = default;

#if _LIBCUDACXX_STD_VER > 17
  constexpr __variant_like(const __variant_like&) requires is_trivially_copy_constructible_v<_Tp>
      && is_trivially_copy_constructible_v<_Up> = default;
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __variant_like(const __variant_like& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>&& is_nothrow_copy_constructible_v<_Up>)
    : __base{__other.__contains_} {
    switch (this->__contains_) {
    case __variant_like_state::__holds_first:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), __other.__first_);
        break;
    case __variant_like_state::__holds_second:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), __other.__second_);
        break;
    case __variant_like_state::__nothing:
        break;
    }
  }

  _LIBCUDACXX_TEMPLATE(class _OTp, class _OUp)
    (requires (!same_as<__variant_like<_OTp, _OUp>, __variant_like>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __variant_like(const __variant_like<_OTp, _OUp>& __other) noexcept(
    is_nothrow_constructible_v<_Tp, const _OTp&> && is_nothrow_constructible_v<_Up, const _OUp&>)
    : __base{__other.__contains_} {
    switch (__other.__contains_) {
    case __variant_like_state::__holds_first:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), __other.__first_);
        break;
    case __variant_like_state::__holds_second:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), __other.__second_);
        break;
    case __variant_like_state::__nothing:
        break;
    }
  }

#if _LIBCUDACXX_STD_VER > 17
  constexpr __variant_like(__variant_like&&) requires is_trivially_move_constructible_v<_Tp>
      && is_trivially_move_constructible_v<_Up> = default;
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __variant_like(__variant_like&& __other) noexcept(
    is_nothrow_move_constructible_v<_Tp>&& is_nothrow_move_constructible_v<_Up>)
    : __base{__other.__contains_} {
    switch (this->__contains_) {
    case __variant_like_state::__holds_first:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), _CUDA_VSTD::move(__other.__first_));
        break;
    case __variant_like_state::__holds_second:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), _CUDA_VSTD::move(__other.__second_));
        break;
    case __variant_like_state::__nothing:
        break;
    }
  }

#if _LIBCUDACXX_STD_VER > 17
  constexpr __variant_like& operator=(const __variant_like&) requires is_trivially_destructible_v<_Tp>
      && is_trivially_destructible_v<_Up>
      && is_trivially_copy_constructible_v<_Tp>
      && is_trivially_copy_constructible_v<_Up>
      && is_trivially_copy_assignable_v<_Tp>
      && is_trivially_copy_assignable_v<_Up> = default;
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __variant_like& operator=(const __variant_like& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>&& is_nothrow_copy_constructible_v<_Up>&&
        is_nothrow_copy_assignable_v<_Tp>&& is_nothrow_copy_assignable_v<_Up>) {
    if (this->__contains_ == __other.__contains_) {
        switch (this->__contains_) {
        case __variant_like_state::__holds_first:
            this->__first_ = __other.__first_;
            break;
        case __variant_like_state::__holds_second:
            this->__second_ = __other.__second_;
            break;
        case __variant_like_state::__nothing:
            break;
        }

        return *this;
    }

    __clear();

    switch (__other.__contains_) {
    case __variant_like_state::__holds_first:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), __other.__first_);
        break;
    case __variant_like_state::__holds_second:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), __other.__second_);
        break;
    case __variant_like_state::__nothing:
        break;
    }

   this->__contains_ = __other.__contains_;

    return *this;
  }

#if _LIBCUDACXX_STD_VER > 17
  constexpr __variant_like& operator=(__variant_like&&) requires is_trivially_destructible_v<_Tp>
      && is_trivially_destructible_v<_Up>
      && is_trivially_move_constructible_v<_Tp>
      && is_trivially_move_constructible_v<_Up>
      && is_trivially_move_assignable_v<_Tp>
      && is_trivially_move_assignable_v<_Up> = default;
#endif

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __variant_like& operator=(__variant_like&& __other) noexcept(
    is_nothrow_move_constructible_v<_Tp>&& is_nothrow_move_constructible_v<_Up>&&
        is_nothrow_move_assignable_v<_Tp>&& is_nothrow_move_assignable_v<_Up>) {
    if (this->__contains_ == __other.__contains_) {
        switch (this->__contains_) {
        case __variant_like_state::__holds_first:
            this->__first_ = _CUDA_VSTD::move(__other.__first_);
            break;
        case __variant_like_state::__holds_second:
            this->__second_ = _CUDA_VSTD::move(__other.__second_);
            break;
        case __variant_like_state::__nothing:
            break;
        }

        return *this;
    }

    __clear();

    switch (__other.__contains_) {
    case __variant_like_state::__holds_first:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), __other.__first_);
        break;
    case __variant_like_state::__holds_second:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), __other.__second_);
        break;
    case __variant_like_state::__nothing:
        break;
    }

   this->__contains_ = __other.__contains_;

    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OTp, class _OUp)
    (requires (!same_as<__variant_like<_OTp, _OUp>, __variant_like>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  __variant_like& operator=(const __variant_like<_OTp, _OUp>& __other) noexcept(
    is_nothrow_constructible_v<_Tp, const _OTp&> && is_nothrow_constructible_v<_Up, const _OUp&>
        && is_nothrow_assignable_v<_Tp&, const _OTp&> && is_nothrow_assignable_v<_Up&, const _OUp&>) {
    if (this->__contains_ == __other.__contains_) {
        switch (this->__contains_) {
        case __variant_like_state::__holds_first:
            this->__first_ = __other.__first_;
            break;
        case __variant_like_state::__holds_second:
            this->__second_ = __other.__second_;
            break;
        case __variant_like_state::__nothing:
            break;
        }

        return *this;
    }

    __clear();

    switch (__other.__contains_) {
    case __variant_like_state::__holds_first:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), __other.__first_);
        break;
    case __variant_like_state::__holds_second:
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), __other.__second_);
        break;
    case __variant_like_state::__nothing:
        break;
    }

   this->__contains_ = __other.__contains_;

    return *this;
  }

  template<class _Tp2 = _Tp>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  friend constexpr auto swap(__variant_like& __x, __variant_like& __y) noexcept(
    is_nothrow_move_constructible_v<_Tp2> && is_nothrow_move_constructible_v<_Up>
        && is_nothrow_swappable_v<_Tp2> && is_nothrow_swappable_v<_Up>)
    _LIBCUDACXX_TRAILING_REQUIRES(void)(requires (!__is_trivially_swappable_v<_Tp2> || !__is_trivially_swappable_v<_Up>)) {
      if (__x.__contains_ == __y.__contains_) {
          switch (__x.__contains_) {
          case __variant_like_state::__holds_first:
              _CUDA_VRANGES::swap(__x.__first_, __y.__first_);
              break;
          case __variant_like_state::__holds_second:
              _CUDA_VRANGES::swap(__x.__second_, __y.__second_);
              break;
          case __variant_like_state::__nothing:
              break;
          }
      } else {
        auto _Tmp = _CUDA_VSTD::move(__x);
        __x       = _CUDA_VSTD::move(__y);
        __y       = _CUDA_VSTD::move(_Tmp);
      }
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  void __clear() noexcept {
    this->__raw_clear();
    this->__contains_ = __variant_like_state::__nothing;
  }

  template <class... _Types>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __emplace_first(_Types&&... _Args) noexcept(is_nothrow_constructible_v<_Tp, _Types...>) {
    this->__raw_clear();
    this->__contains_ = __variant_like_state::__nothing;

    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__first_), _CUDA_VSTD::forward<_Types>(_Args)...);
    this->__contains_ = __variant_like_state::__holds_first;
  }

  template <class... _Types>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
  void __emplace_second(_Types&&... _Args) noexcept(is_nothrow_constructible_v<_Up, _Types...>) {
    this->__raw_clear();
    this->__contains_ = __variant_like_state::__nothing;

    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__second_), _CUDA_VSTD::forward<_Types>(_Args)...);
    this->__contains_ = __variant_like_state::__holds_second;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool valueless_by_exception() const noexcept {
    return this->__contains_ == __variant_like_state::__nothing;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool __holds_first() const noexcept {
    return this->__contains_ == __variant_like_state::__holds_first;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr
  bool __holds_second() const noexcept {
    return this->__contains_ == __variant_like_state::__holds_second;
  }
};

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#pragma warning(pop)
#endif // _LIBCUDACXX_COMPILER_MSVC

#endif // _LIBCUDACXX___RANGES_VARIANT_LIKE_H
