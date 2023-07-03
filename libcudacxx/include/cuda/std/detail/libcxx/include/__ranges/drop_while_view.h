// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_DROP_WHILE_VIEW_H
#define _LIBCUDACXX___RANGES_DROP_WHILE_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__algorithm/ranges_find_if_not.h"
#include "../__concepts/constructible.h"
#include "../__debug"
#include "../__functional/bind_back.h"
#include "../__functional/reference_wrapper.h"
#include "../__iterator/concepts.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/copyable_box.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/non_propagating_cache.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/is_nothrow_constructible.h"
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
template <view _View, class _Pred>
  requires input_range<_View> && is_object_v<_Pred> && indirect_unary_predicate<const _Pred, iterator_t<_View>>
#else
  template<class _View, class _Pred,
           class = enable_if_t<view<_View>>,
           class = enable_if_t<input_range<_View>>,
           class = enable_if_t<is_object_v<_Pred>>,
           class = enable_if_t<indirect_unary_predicate<const _Pred, iterator_t<_View>>>>
#endif // _LIBCUDACXX_STD_VER < 20
class drop_while_view : public view_interface<drop_while_view<_View, _Pred>> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __copyable_box<_Pred> __pred_;

  static constexpr bool _UseCache = forward_range<_View>;
  using _Cache                    = _If<_UseCache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();

public:
#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  drop_while_view() requires default_initializable<_View> && default_initializable<_Pred> = default;
#else
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
      (requires default_initializable<_View2> && default_initializable<_Pred>)
    _LIBCUDACXX_INLINE_VISIBILITY
    constexpr drop_while_view() noexcept(is_nothrow_default_constructible_v<_View2>
                                      && is_nothrow_default_constructible_v<_Pred>)
      : view_interface<drop_while_view<_View, _Pred>>() {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_INLINE_VISIBILITY constexpr drop_while_view(_View __base, _Pred __pred)
      : view_interface<drop_while_view<_View, _Pred>>()
      , __base_(_CUDA_VSTD::move(__base)), __pred_(in_place, _CUDA_VSTD::move(__pred))
    { }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_INLINE_VISIBILITY constexpr _View base() const&
  {
    return __base_;
  }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr _View base() && { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr const _Pred& pred() const { return *__pred_; }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr auto begin() {
    _LIBCUDACXX_ASSERT(__pred_.__has_value(),
                   "drop_while_view needs to have a non-empty predicate before calling begin() -- did a previous "
                   "assignment to this drop_while_view fail?");
    if constexpr (_UseCache) {
      if (!__cached_begin_.__has_value()) {
        __cached_begin_.__emplace(_CUDA_VRANGES::find_if_not(__base_, _CUDA_VSTD::cref(*__pred_)));
      }
      return *__cached_begin_;
    } else {
      return _CUDA_VRANGES::find_if_not(__base_, _CUDA_VSTD::cref(*__pred_));
    }
    _LIBCUDACXX_UNREACHABLE();
  }

  _LIBCUDACXX_INLINE_VISIBILITY constexpr auto end() { return _CUDA_VRANGES::end(__base_); }
};

template <class _Range, class _Pred>
drop_while_view(_Range&&, _Pred) -> drop_while_view<_CUDA_VIEWS::all_t<_Range>, _Pred>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _View, class _Pred>
inline constexpr bool enable_borrowed_range<drop_while_view<_View, _Pred>> = enable_borrowed_range<_View>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__drop_while)

struct __fn {
  template <class _Range, class _Pred>
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Range&& __range, _Pred&& __pred) const
      noexcept(noexcept(/**/ drop_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))))
          -> decltype(/*--*/ drop_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))) {
    return /*-------------*/ drop_while_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred));
  }

  _LIBCUDACXX_TEMPLATE(class _Pred)
    (requires constructible_from<decay_t<_Pred>, _Pred>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Pred&& __pred) const
      noexcept(is_nothrow_constructible_v<decay_t<_Pred>, _Pred>) {
    return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Pred>(__pred)));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto drop_while = __drop_while::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_DROP_WHILE_VIEW_H
