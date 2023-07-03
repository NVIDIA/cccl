// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_TAKE_WHILE_VIEW_H
#define _LIBCUDACXX___RANGES_TAKE_WHILE_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__functional/bind_back.h"
#include "../__functional/invoke.h"
#include "../__iterator/concepts.h"
#include "../__memory/addressof.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/copyable_box.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/maybe_const.h"
#include "../__utility/forward.h"
#include "../__utility/in_place.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 17
template <class _View, class _Pred>
concept __take_while_const_is_range =
    range<const _View> && indirect_unary_predicate<const _Pred, iterator_t<const _View>>;
#else
template <class _View, class _Pred>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __take_while_const_is_range_,
    requires()(
      requires(range<const _View>),
      requires(indirect_unary_predicate<const _Pred, iterator_t<const _View>>)
    ));

template <class _View, class _Pred>
  _LIBCUDACXX_CONCEPT __take_while_const_is_range = _LIBCUDACXX_FRAGMENT(__take_while_const_is_range_, _View, _Pred);
#endif

#if _LIBCUDACXX_STD_VER > 17
template <view _View, class _Pred>
  requires input_range<_View> && is_object_v<_Pred> && indirect_unary_predicate<const _Pred, iterator_t<_View>>
#else
template<class _View, class _Pred,
         class = enable_if_t<view<_View>>,
         class = enable_if_t<input_range<_View>>,
         class = enable_if_t<is_object_v<_Pred>>,
         class = enable_if_t<indirect_unary_predicate<const _Pred, iterator_t<_View>>>>
#endif // _LIBCUDACXX_STD_VER < 20
class take_while_view : public view_interface<take_while_view<_View, _Pred>> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __copyable_box<_Pred> __pred_;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();

public:
  template <bool _Const>
  class __sentinel {
    using _Base = __maybe_const<_Const, _View>;

    sentinel_t<_Base> __end_ = sentinel_t<_Base>();
    const _Pred* __pred_     = nullptr;

    friend class __sentinel<!_Const>;

  public:
    _LIBCUDACXX_HIDE_FROM_ABI __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit __sentinel(sentinel_t<_Base> __end, const _Pred* __pred)
        : __end_(_CUDA_VSTD::move(__end)), __pred_(__pred) {}

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      (requires _OtherConst _LIBCUDACXX_AND convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr __sentinel(__sentinel<!_OtherConst> __s)
        : __end_(_CUDA_VSTD::move(__s.__end_)), __pred_(__s.__pred_) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr sentinel_t<_Base> base() const { return __end_; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator==(const iterator_t<_Base>& __x, const __sentinel& __y) {
      return __x == __y.__end_ || !_CUDA_VSTD::invoke(*__y.__pred_, *__x);
    }
  #if _LIBCUDACXX_STD_VER < 20
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator==(const __sentinel& __x, const iterator_t<_Base>& __y) {
      return __y == __x.__end_ || !_CUDA_VSTD::invoke(*__x.__pred_, *__y);
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator!=(const iterator_t<_Base>& __x, const __sentinel& __y) {
      return __x != __y.__end_ && _CUDA_VSTD::invoke(*__y.__pred_, *__x);
    }
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY friend constexpr bool operator!=(const __sentinel& __x, const iterator_t<_Base>& __y) {
      return __y != __x.__end_ && _CUDA_VSTD::invoke(*__x.__pred_, *__y);
    }
  #endif // _LIBCUDACXX_STD_VER < 20

    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const iterator_t<__maybe_const<_OtherConst, _View>>& __x, const __sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __x == __y.__end_ || !_CUDA_VSTD::invoke(*__y.__pred_, *__x);
    }
  #if _LIBCUDACXX_STD_VER < 20
    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __sentinel& __x, const iterator_t<__maybe_const<_OtherConst, _View>>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __y == __x.__end_ || !_CUDA_VSTD::invoke(*__x.__pred_, *__y);
    }
    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const iterator_t<__maybe_const<_OtherConst, _View>>& __x, const __sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __x != __y.__end_ && _CUDA_VSTD::invoke(*__y.__pred_, *__x);
    }
    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __sentinel& __x, const iterator_t<__maybe_const<_OtherConst, _View>>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __y != __x.__end_ && _CUDA_VSTD::invoke(*__x.__pred_, *__y);
    }
  #endif // _LIBCUDACXX_STD_VER < 20
  };

#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI take_while_view()
    requires default_initializable<_View> && default_initializable<_Pred>
  = default;
#else
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires default_initializable<_View2> _LIBCUDACXX_AND default_initializable<_Pred>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr take_while_view() noexcept(
    is_nothrow_default_constructible_v<_View2> && is_nothrow_default_constructible_v<_Pred>) {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr take_while_view(_View __base, _Pred __pred)
      : view_interface<take_while_view<_View, _Pred>>()
      , __pred_(_CUDA_VSTD::in_place, _CUDA_VSTD::move(__pred))
      , __base_(_CUDA_VSTD::move(__base)) {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _View base() const&
  {
    return __base_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _View base() && { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Pred& pred() const { return *__pred_; }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!__simple_view<_View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto begin()
  {
    return _CUDA_VRANGES::begin(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires __take_while_const_is_range<_View2, _Pred>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto begin() const

  {
    return _CUDA_VRANGES::begin(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!__simple_view<_View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end()
  {
    return __sentinel</*_Const=*/false>(_CUDA_VRANGES::end(__base_), _CUDA_VSTD::addressof(*__pred_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires __take_while_const_is_range<_View2, _Pred>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end() const
  {
    return __sentinel</*_Const=*/true>(_CUDA_VRANGES::end(__base_), _CUDA_VSTD::addressof(*__pred_));
  }
};

template <class _Range, class _Pred>
take_while_view(_Range&&, _Pred) -> take_while_view<_CUDA_VIEWS::all_t<_Range>, _Pred>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__take_while)

struct __fn {
  template <class _Range, class _Pred>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range, _Pred&& __pred) const
      noexcept(noexcept(/**/ take_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))))
          -> decltype(/*--*/ take_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))) {
    return /*-------------*/ take_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred));
  }

  _LIBCUDACXX_TEMPLATE(class _Pred)
    (requires constructible_from<decay_t<_Pred>, _Pred>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Pred&& __pred) const
      noexcept(is_nothrow_constructible_v<decay_t<_Pred>, _Pred>) {
    return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Pred>(__pred)));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto take_while = __take_while::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_TAKE_WHILE_VIEW_H
