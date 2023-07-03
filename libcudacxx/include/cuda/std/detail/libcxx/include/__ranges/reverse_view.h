// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_REVERSE_VIEW_H
#define _LIBCUDACXX___RANGES_REVERSE_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__iterator/concepts.h"
#include "../__iterator/next.h"
#include "../__iterator/reverse_iterator.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/non_propagating_cache.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/size.h"
#include "../__ranges/subrange.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_nothrow_copy_constructible.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif


#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#if _LIBCUDACXX_STD_VER > 17
template<view _View>
  requires bidirectional_range<_View>
#else
template<class _View, class = enable_if_t<view<_View>>
                    , class = enable_if_t<bidirectional_range<_View>>>
#endif
class reverse_view : public view_interface<reverse_view<_View>> {
  // We cache begin() whenever ranges::next is not guaranteed O(1) to provide an
  // amortized O(1) begin() method.
  static constexpr bool _UseCache = !random_access_range<_View> && !common_range<_View>;
  using _Cache = _If<_UseCache, __non_propagating_cache<reverse_iterator<iterator_t<_View>>>, __empty_cache>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();

public:
#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  reverse_view() requires default_initializable<_View> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires default_initializable<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr reverse_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<reverse_view<_View>>() {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr explicit reverse_view(_View __view) : __base_(_CUDA_VSTD::move(__view)) {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View>) { return __base_; }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>) { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!common_range<_View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr reverse_iterator<iterator_t<_View2>> begin() {
    if constexpr (_UseCache)
      if (__cached_begin_.__has_value())
        return *__cached_begin_;

    auto __tmp = _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::next(_CUDA_VRANGES::begin(__base_), _CUDA_VRANGES::end(__base_)));
    if constexpr (_UseCache)
      __cached_begin_.__emplace(__tmp);
    return __tmp;
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires common_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr reverse_iterator<iterator_t<_View2>> begin() {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::end(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires common_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto begin() const {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::end(__base_));
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr reverse_iterator<iterator_t<_View>> end() {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires common_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto end() const {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() {
    return _CUDA_VRANGES::size(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() const {
    return _CUDA_VRANGES::size(__base_);
  }
};

template<class _Range>
reverse_view(_Range&&) -> reverse_view<_CUDA_VIEWS::all_t<_Range>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template<class _Tp>
inline constexpr bool enable_borrowed_range<reverse_view<_Tp>> = enable_borrowed_range<_Tp>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__reverse)

template<class _Tp>
inline constexpr bool __is_reverse_view = false;

template<class _Tp>
inline constexpr bool __is_reverse_view<reverse_view<_Tp>> = true;

template<class _Tp>
inline constexpr bool __is_sized_reverse_subrange = false;

template<class _Iter>
inline constexpr bool __is_sized_reverse_subrange<_CUDA_VRANGES::subrange<reverse_iterator<_Iter>, reverse_iterator<_Iter>, _CUDA_VRANGES::subrange_kind::sized>> = true;

template<class _Tp>
inline constexpr bool __is_unsized_reverse_subrange = false;

template<class _Iter, subrange_kind _Kind>
inline constexpr bool __is_unsized_reverse_subrange<_CUDA_VRANGES::subrange<reverse_iterator<_Iter>, reverse_iterator<_Iter>, _Kind>> = _Kind == _CUDA_VRANGES::subrange_kind::unsized;

template<class _Tp>
struct __unwrapped_reverse_subrange {
  using type = void; // avoid SFINAE-ing out the overload below -- let the concept requirements do it for better diagnostics
};

template<class _Iter, _CUDA_VRANGES::subrange_kind _Kind>
struct __unwrapped_reverse_subrange<_CUDA_VRANGES::subrange<reverse_iterator<_Iter>, reverse_iterator<_Iter>, _Kind>> {
  using type = _CUDA_VRANGES::subrange<_Iter, _Iter, _Kind>;
};

template<class _Tp>
using __unwrapped_reverse_subrange_t = typename __unwrapped_reverse_subrange<_Tp>::type;

struct __fn : __range_adaptor_closure<__fn> {
  _LIBCUDACXX_TEMPLATE(class _Range)
    (requires __is_reverse_view<remove_cvref_t<_Range>>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(_CUDA_VSTD::forward<_Range>(__range).base()))
    -> decltype(      _CUDA_VSTD::forward<_Range>(__range).base())
    { return          _CUDA_VSTD::forward<_Range>(__range).base(); }

  _LIBCUDACXX_TEMPLATE(class _Range, class _UnwrappedSubrange = __unwrapped_reverse_subrange_t<remove_cvref_t<_Range>>)
    (requires __is_sized_reverse_subrange<remove_cvref_t<_Range>>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(_UnwrappedSubrange(__range.end().base(), __range.begin().base(), __range.size())))
    -> decltype(      _UnwrappedSubrange(__range.end().base(), __range.begin().base(), __range.size()))
    { return          _UnwrappedSubrange(__range.end().base(), __range.begin().base(), __range.size()); }


  _LIBCUDACXX_TEMPLATE(class _Range, class _UnwrappedSubrange = __unwrapped_reverse_subrange_t<remove_cvref_t<_Range>>)
    (requires __is_unsized_reverse_subrange<remove_cvref_t<_Range>>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(_UnwrappedSubrange(__range.end().base(), __range.begin().base())))
    -> decltype(      _UnwrappedSubrange(__range.end().base(), __range.begin().base()))
    { return          _UnwrappedSubrange(__range.end().base(), __range.begin().base()); }

  _LIBCUDACXX_TEMPLATE(class _Range)
    (requires (!__is_reverse_view<remove_cvref_t<_Range>>) _LIBCUDACXX_AND
              (!__is_sized_reverse_subrange<remove_cvref_t<_Range>>) _LIBCUDACXX_AND
              (!__is_unsized_reverse_subrange<remove_cvref_t<_Range>>))
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(reverse_view{_CUDA_VSTD::forward<_Range>(__range)}))
    -> decltype(      reverse_view{_CUDA_VSTD::forward<_Range>(__range)})
    { return          reverse_view{_CUDA_VSTD::forward<_Range>(__range)}; }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto reverse = __reverse::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_VIEWS


#endif // _LIBCUDACXX_STD_VER > 17


#endif // _LIBCUDACXX___RANGES_REVERSE_VIEW_H
