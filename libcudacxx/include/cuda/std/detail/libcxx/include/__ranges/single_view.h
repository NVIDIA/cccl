// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_SINGLE_VIEW_H
#define _LIBCUDACXX___RANGES_SINGLE_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__ranges/copyable_box.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_object.h"
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
  template<copy_constructible _Tp>
    requires is_object_v<_Tp>
#else
  template<class _Tp, enable_if_t<copy_constructible<_Tp>, int> = 0,
                      enable_if_t<is_object_v<_Tp>, int> = 0>
#endif
  class single_view : public view_interface<single_view<_Tp>> {
    __copyable_box<_Tp> __value_;

  public:
#if _LIBCUDACXX_STD_VER > 17
    _LIBCUDACXX_HIDE_FROM_ABI
    single_view() requires default_initializable<_Tp> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
      (requires default_initializable<_Tp2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr single_view() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>(), __value_() {};
#endif

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit single_view(const _Tp& __t) noexcept(is_nothrow_copy_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>(), __value_(in_place, __t) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit single_view(_Tp&& __t) noexcept(is_nothrow_move_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>(), __value_(in_place, _CUDA_VSTD::move(__t)) {}

    _LIBCUDACXX_TEMPLATE(class... _Args)
      (requires constructible_from<_Tp, _Args...>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit single_view(in_place_t, _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : view_interface<single_view<_Tp>>(), __value_{in_place, _CUDA_VSTD::forward<_Args>(__args)...} {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _Tp* begin() noexcept { return data(); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const _Tp* begin() const noexcept { return data(); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _Tp* end() noexcept { return data() + 1; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const _Tp* end() const noexcept { return data() + 1; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    static constexpr size_t size() noexcept { return 1; }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr _Tp* data() noexcept { return __value_.operator->(); }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const _Tp* data() const noexcept { return __value_.operator->(); }
  };

template<class _Tp>
single_view(_Tp) -> single_view<_Tp>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__single_view)

template<class _Range>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __can_single_view_,
  requires(_Range&& __range)(
    typename(single_view<decay_t<_Range>>)
  ));

template<class _Range>
_LIBCUDACXX_CONCEPT __can_single_view = _LIBCUDACXX_FRAGMENT(__can_single_view_, _Range);

struct __fn : __range_adaptor_closure<__fn> {
  _LIBCUDACXX_TEMPLATE(class _Range)
    (requires __can_single_view<_Range>) // MSVC breaks without it
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(single_view<decay_t<_Range>>(_CUDA_VSTD::forward<_Range>(__range))))
    -> decltype(      single_view<decay_t<_Range>>(_CUDA_VSTD::forward<_Range>(__range)))
    { return          single_view<decay_t<_Range>>(_CUDA_VSTD::forward<_Range>(__range)); }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto single = __single_view::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_SINGLE_VIEW_H
