// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_TRANSFORM_VIEW_H
#define _LIBCUDACXX___RANGES_TRANSFORM_VIEW_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include "../__compare/three_way_comparable.h"
#endif
#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/copyable.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/equality_comparable.h"
#include "../__concepts/invocable.h"
#include "../__functional/bind_back.h"
#include "../__functional/invoke.h"
#include "../__iterator/concepts.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../__ranges/access.h"
#include "../__ranges/all.h"
#include "../__ranges/concepts.h"
#include "../__ranges/copyable_box.h"
#include "../__ranges/empty.h"
#include "../__ranges/range_adaptor.h"
#include "../__ranges/size.h"
#include "../__ranges/view_interface.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_reference.h"
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

  template<class _Fn, class _View>
  _LIBCUDACXX_CONCEPT __regular_invocable_with_range_ref =
    regular_invocable<_Fn, range_reference_t<_View>>;

#if _LIBCUDACXX_STD_VER > 17
  template<class _View, class _Fn>
  concept __transform_view_constraints =
    view<_View> && is_object_v<_Fn> &&
    regular_invocable<_Fn&, range_reference_t<_View>> &&
    __can_reference<invoke_result_t<_Fn&, range_reference_t<_View>>>;
#else
  template<class _View, class _Fn>
  _LIBCUDACXX_CONCEPT_FRAGMENT(
    __transform_view_constraints_,
    requires()(
      requires(view<_View>),
      requires(is_object_v<_Fn>),
      requires(regular_invocable<_Fn&, range_reference_t<_View>>),
      requires(__can_reference<invoke_result_t<_Fn&, range_reference_t<_View>>>)
    ));

  template<class _View, class _Fn>
  _LIBCUDACXX_CONCEPT __transform_view_constraints = _LIBCUDACXX_FRAGMENT(__transform_view_constraints_, _View, _Fn);
#endif // _LIBCUDACXX_STD_VER < 20

template<class, class, class = void>
struct __transform_view_iterator_category_base {};

template<class _View, class _Fn>
struct __transform_view_iterator_category_base<_View, _Fn, enable_if_t<forward_range<_View>>> {
  using _Cat = typename iterator_traits<iterator_t<_View>>::iterator_category;

  using iterator_category = conditional_t<
    is_reference_v<invoke_result_t<_Fn&, range_reference_t<_View>>>,
    conditional_t<
      derived_from<_Cat, contiguous_iterator_tag>,
      random_access_iterator_tag,
      _Cat
    >,
    input_iterator_tag
  >;
};

template<class _Fn, class _Iter, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool __nothrow_subscript = false;

template<class _Fn, class _Iter>
_LIBCUDACXX_INLINE_VAR constexpr bool __nothrow_subscript<_Fn, _Iter, enable_if_t<random_access_iterator<_Iter>>>
  = noexcept(_CUDA_VSTD::invoke(_CUDA_VSTD::declval<const _Fn&>(), _CUDA_VSTD::declval<const _Iter&>()[0]));

#if _LIBCUDACXX_STD_VER > 17
template<input_range _View, copy_constructible _Fn>
  requires __transform_view_constraints<_View, _Fn>
#else
template<class _View, class _Fn,
         class = enable_if_t<input_range<_View>>,
         class = enable_if_t<copy_constructible<_Fn>>,
         class = enable_if_t<__transform_view_constraints<_View, _Fn>>>
#endif // _LIBCUDACXX_STD_VER < 20
class transform_view : public view_interface<transform_view<_View, _Fn>> {
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _View __base_ = _View();
  _LIBCUDACXX_NO_UNIQUE_ADDRESS __copyable_box<_Fn> __func_;

public:
  template<bool>
  class __sentinel;

  template<bool _Const>
  class __iterator
    : public __transform_view_iterator_category_base<_View, _Fn> {

    using _Parent = __maybe_const<_Const, transform_view>;
    using _Base = __maybe_const<_Const, _View>;

    _Parent *__parent_ = nullptr;

    template<bool>
    friend class __iterator;

    template<bool>
    friend class __sentinel;

  public:
    iterator_t<_Base> __current_ = iterator_t<_Base>();

    using iterator_concept = conditional_t<random_access_range<_Base>, random_access_iterator_tag,
                            conditional_t<bidirectional_range<_Base>, bidirectional_iterator_tag,
                            conditional_t<forward_range<_Base>, forward_iterator_tag, input_iterator_tag>>>;
    using value_type = remove_cvref_t<invoke_result_t<__maybe_const<_Const, _Fn>&, range_reference_t<_Base>>>;
    using difference_type = range_difference_t<_Base>;

  #if _LIBCUDACXX_STD_VER > 17
    _LIBCUDACXX_HIDE_FROM_ABI
    __iterator() requires default_initializable<iterator_t<_Base>> = default;
  #else
    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires default_initializable<iterator_t<_Base2>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator() noexcept(is_nothrow_default_constructible_v<_Base2>) {}
  #endif // _LIBCUDACXX_STD_VER < 20

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator(_Parent& __parent, iterator_t<_Base> __current)
      : __parent_(_CUDA_VSTD::addressof(__parent)), __current_(_CUDA_VSTD::move(__current)) {}

    // Note: `__i` should always be `__iterator<false>`, but directly using
    // `__iterator<false>` is ill-formed when `_Const` is false
    // (see http://wg21.link/class.copy.ctor#5).
    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      (requires _OtherConst _LIBCUDACXX_AND convertible_to<iterator_t<_View>, iterator_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator(__iterator<!_OtherConst> __i)
      : __parent_(__i.__parent_), __current_(_CUDA_VSTD::move(__i.__current_)) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr const iterator_t<_Base>& base() const& noexcept {
      return __current_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr iterator_t<_Base> base() && {
      return _CUDA_VSTD::move(__current_);
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator*() const
      noexcept(noexcept(_CUDA_VSTD::invoke(*__parent_->__func_, *__current_)))
    {
      return _CUDA_VSTD::invoke(*__parent_->__func_, *__current_);
    }

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator++() {
      ++__current_;
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires (!forward_range<_Base2>))
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr void operator++(int) { ++__current_; }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires forward_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator operator++(int)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires bidirectional_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator--()
    {
      --__current_;
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires bidirectional_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator operator--(int)
    {
      auto __tmp = *this;
      --*this;
      return __tmp;
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires random_access_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator+=(difference_type __n)
    {
      __current_ += __n;
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires random_access_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __iterator& operator-=(difference_type __n)
    {
      __current_ -= __n;
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires random_access_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr decltype(auto) operator[](difference_type __n) const
      noexcept(__nothrow_subscript<_Fn, _Base2>)
    {
      return _CUDA_VSTD::invoke(*__parent_->__func_, __current_[__n]);
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires equality_comparable<iterator_t<_Base2>>)
    {
      return __x.__current_ == __y.__current_;
    }
  #if _LIBCUDACXX_STD_VER < 20
    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires equality_comparable<iterator_t<_Base2>>)
    {
      return __x.__current_ != __y.__current_;
    }
  #endif

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator<(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2>)
    {
      return __x.__current_ < __y.__current_;
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator>(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2>)
    {
      return __x.__current_ > __y.__current_;
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator<=(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2>)
    {
      return __x.__current_ <= __y.__current_;
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator>=(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(requires random_access_range<_Base2>)
    {
      return __x.__current_ >= __y.__current_;
    }

  #ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
      (requires random_access_range<_Base2> _LIBCUDACXX_AND three_way_comparable<iterator_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    {
      return __x.__current_ <=> __y.__current_;
    }
  #endif

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator+(__iterator __i, difference_type __n)
      _LIBCUDACXX_TRAILING_REQUIRES(__iterator)(requires random_access_range<_Base2>)
    {
      return __iterator{*__i.__parent_, __i.__current_ + __n};
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator+(difference_type __n, __iterator __i)
      _LIBCUDACXX_TRAILING_REQUIRES(__iterator)(requires random_access_range<_Base2>)
    {
      return __iterator{*__i.__parent_, __i.__current_ + __n};
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator-(__iterator __i, difference_type __n)
      _LIBCUDACXX_TRAILING_REQUIRES(__iterator)(requires random_access_range<_Base2>)
    {
      return __iterator{*__i.__parent_, __i.__current_ - __n};
    }

    template<class _Base2 = _Base>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator-(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(difference_type)(requires sized_sentinel_for<iterator_t<_Base2>, iterator_t<_Base2>>)
    {
      return __x.__current_ - __y.__current_;
    }
  };

  template<bool _Const>
  class __sentinel {
    using _Parent = __maybe_const<_Const, transform_view>;
    using _Base = __maybe_const<_Const, _View>;

    sentinel_t<_Base> __end_ = sentinel_t<_Base>();

    template<bool>
    friend class __iterator;

    template<bool>
    friend class __sentinel;

  public:
    _LIBCUDACXX_HIDE_FROM_ABI
    __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr explicit __sentinel(sentinel_t<_Base> __end) : __end_(__end) {}

    // Note: `__i` should always be `__sentinel<false>`, but directly using
    // `__sentinel<false>` is ill-formed when `_Const` is false
    // (see http://wg21.link/class.copy.ctor#5).
    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
      (requires _OtherConst && convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr __sentinel(__sentinel<!_OtherConst> __i)
      : __end_(_CUDA_VSTD::move(__i.__end_)) {}

    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr sentinel_t<_Base> base() const { return __end_; }

    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
        (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__current_ == __y.__end_;
    }
#if _LIBCUDACXX_STD_VER < 20
    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator==(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
        (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__end_ == __y.__current_;
    }

    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
        (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__current_ != __y.__end_;
    }

    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator!=(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)
        (requires sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__end_ != __y.__current_;
    }
#endif

    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator-(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(range_difference_t<__maybe_const<_OtherConst, _View>>)
        (requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__current_ - __y.__end_;
    }

    template<bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator-(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(range_difference_t<__maybe_const<_OtherConst, _View>>)
        (requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>) {
      return __x.__end_ - __y.__current_;
    }
  };

#if _LIBCUDACXX_STD_VER > 17
  _LIBCUDACXX_HIDE_FROM_ABI
  transform_view()
    requires default_initializable<_View> && default_initializable<_Fn> = default;
#else
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires default_initializable<_View2> && default_initializable<_Fn>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr transform_view() noexcept(is_nothrow_default_constructible_v<_View2> && is_nothrow_default_constructible_v<_Fn>)
  {}
#endif // _LIBCUDACXX_STD_VER < 20

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr transform_view(_View __base, _Fn __func)
    : view_interface<transform_view<_View, _Fn>>()
    , __base_(_CUDA_VSTD::move(__base))
    , __func_(_CUDA_VSTD::in_place, _CUDA_VSTD::move(__func))
  {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() const& { return __base_; }
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _View base() && { return _CUDA_VSTD::move(__base_); }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __iterator<false> begin() {
    return __iterator<false>{*this, _CUDA_VRANGES::begin(__base_)};
  }
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires range<const _View2> _LIBCUDACXX_AND
             __regular_invocable_with_range_ref<const _Fn&, const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __iterator<true> begin() const
  {
    return __iterator<true>(*this, _CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (!common_range<_View2>))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __sentinel<false> end() {
    return __sentinel<false>(_CUDA_VRANGES::end(__base_));
  }
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires common_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __iterator<false> end()
  {
    return __iterator<false>(*this, _CUDA_VRANGES::end(__base_));
  }
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires (range<const _View2> && !common_range<const _View2>) _LIBCUDACXX_AND
             __regular_invocable_with_range_ref<const _Fn&, const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __sentinel<true> end() const
  {
    return __sentinel<true>(_CUDA_VRANGES::end(__base_));
  }
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires common_range<const _View2> _LIBCUDACXX_AND
             __regular_invocable_with_range_ref<const _Fn&, const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr __iterator<true> end() const
  {
    return __iterator<true>(*this, _CUDA_VRANGES::end(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() { return _CUDA_VRANGES::size(__base_); }
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    (requires sized_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto size() const { return _CUDA_VRANGES::size(__base_); }
};

template<class _Range, class _Fn>
transform_view(_Range&&, _Fn) -> transform_view<_CUDA_VIEWS::all_t<_Range>, _Fn>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__transform)
  struct __fn {
    template<class _Range, class _Fn>
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Range&& __range, _Fn&& __f) const
      noexcept(noexcept(transform_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Fn>(__f))))
      -> decltype(      transform_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Fn>(__f)))
      { return          transform_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Fn>(__f)); }

    _LIBCUDACXX_TEMPLATE(class _Fn)
      (requires constructible_from<decay_t<_Fn>, _Fn>)
    _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    constexpr auto operator()(_Fn&& __f) const
      noexcept(is_nothrow_constructible_v<decay_t<_Fn>, _Fn>)
    { return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Fn>(__f))); }
  };
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto transform = __transform::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_VIEWS

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___RANGES_TRANSFORM_VIEW_H
