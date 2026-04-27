// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_FILTER_VIEW_H
#define _CUDA_STD___RANGES_FILTER_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/ranges_find_if.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__functional/bind_back.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/reference_wrapper.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

template <class _View, bool = forward_range<_View>>
struct __filter_iterator_category
{};

template <class _View>
struct __filter_iterator_category<_View, true>
{
  using category _CCCL_NODEBUG = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using iterator_category =
    conditional_t<derived_from<category, bidirectional_iterator_tag>,
                  bidirectional_iterator_tag,
                  conditional_t<derived_from<category, forward_iterator_tag>,
                                forward_iterator_tag,
                                /* else */ category>>;
};

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

#if _CCCL_HAS_CONCEPTS()
template <input_range _View, indirect_unary_predicate<iterator_t<_View>> _Pred>
  requires view<_View> && is_object_v<_Pred>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _View,
          class _Pred,
          enable_if_t<input_range<_View>, int>                                 = 0,
          enable_if_t<view<_View>, int>                                        = 0,
          enable_if_t<indirect_unary_predicate<_Pred, iterator_t<_View>>, int> = 0,
          enable_if_t<is_object_v<_Pred>, int>                                 = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class filter_view : public view_interface<filter_view<_View, _Pred>>
{
  _View __base_;
  __movable_box<_Pred> __pred_;

  // We cache the result of begin() to allow providing an amortized O(1) begin() whenever
  // the underlying range is at least a forward_range.
  static constexpr bool __use_cache = forward_range<_View>;
  using _cache_type _CCCL_NODEBUG =
    conditional_t<__use_cache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _cache_type __cached_begin_;

public:
  // __iterator and __sentinel should be private, but several compilers (clang-14, gcc-12,
  // msvc-19.39) complain that we are naming a private type when we say decltype(view.end()),
  // but that is wrong. You are allowed to decltype (or auto) private types, even if you cannot
  // spell them out explicitly. This also seems to fire when calling operator==(), so the only
  // workaround is to make these classes public for those compilers.
  class __iterator : public __filter_iterator_category<_View>
  {
  public:
    iterator_t<_View> __current_;
    filter_view* __parent_ = nullptr;

    using iterator_concept =
      conditional_t<bidirectional_range<_View>,
                    bidirectional_iterator_tag,
                    conditional_t<forward_range<_View>,
                                  forward_iterator_tag,
                                  /* else */ input_iterator_tag>>;

    // using iterator_category = inherited;
    using value_type      = range_value_t<_View>;
    using difference_type = range_difference_t<_View>;

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(default_initializable<iterator_t<_View2>>)
    _CCCL_API constexpr __iterator() noexcept(is_nothrow_default_constructible_v<iterator_t<_View2>>)
        : __current_()
    {}

    _CCCL_API constexpr __iterator(filter_view& __parent, iterator_t<_View> __current) noexcept(
      is_nothrow_move_constructible_v<iterator_t<_View>>)
        : __current_{::cuda::std::move(__current)}
        , __parent_{::cuda::std::addressof(__parent)}
    {}

    [[nodiscard]] _CCCL_API constexpr const iterator_t<_View>& base() const& noexcept
    {
      return __current_;
    }

    [[nodiscard]] _CCCL_API constexpr iterator_t<_View>
    base() && noexcept(is_nothrow_move_constructible_v<iterator_t<_View>>)
    {
      return ::cuda::std::move(__current_);
    }

    [[nodiscard]] _CCCL_API constexpr range_reference_t<_View> operator*() const
      noexcept(is_nothrow_constructible_v<range_reference_t<_View>, decltype(*__current_)>)
    {
      return *__current_;
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(__has_arrow<iterator_t<_View2>> _CCCL_AND copyable<iterator_t<_View2>>)
    _CCCL_API constexpr iterator_t<_View> operator->() const
      noexcept(is_nothrow_copy_constructible_v<iterator_t<_View2>>)
    {
      return __current_;
    }

    _CCCL_API constexpr __iterator& operator++()
    {
      __current_ = ::cuda::std::ranges::find_if(
        ::cuda::std::move(++__current_),
        ::cuda::std::ranges::end(__parent_->__base_),
        ::cuda::std::ref(*__parent_->__pred_));
      return *this;
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES((!forward_range<_View2>) )
    _CCCL_API constexpr void operator++(int)
    {
      ++*this;
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(forward_range<_View2>)
    _CCCL_API constexpr __iterator operator++(int)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(bidirectional_range<_View2>)
    _CCCL_API constexpr __iterator& operator--()
    {
      do
      {
        --__current_;
      } while (!::cuda::std::invoke(*__parent_->__pred_, *__current_));
      return *this;
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(bidirectional_range<_View2>)
    _CCCL_API constexpr __iterator operator--(int)
    {
      auto __tmp = *this;
      --*this;
      return __tmp;
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(equality_comparable<iterator_t<_View2>>)
    _CCCL_API friend constexpr bool operator==(const __iterator& __x, const __iterator& __y) noexcept(
      __is_cpp17_nothrow_equality_comparable_v<iterator_t<_View>, iterator_t<_View>>)
    {
      return __x.__current_ == __y.__current_;
    }

#if _CCCL_STD_VER <= 2017
    // Cannot do noexcept(noexcept(__x == __y)) because __iterator may not even be comparable,
    // so the above would be an ill-formed expression. The type-trait handles this case transparently.
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __iterator& __x, const __iterator& __y) noexcept(
      __is_cpp17_nothrow_equality_comparable_v<__iterator, __iterator>)
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    [[nodiscard]] _CCCL_API friend constexpr auto
    operator<=>(const __iterator& __x, const __iterator& __y) noexcept(noexcept(__x.__current_ <=> __y.__current_))
    {
      return __x.__current_ <=> __y.__current_;
    }
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR()

    [[nodiscard]] _CCCL_API friend constexpr range_rvalue_reference_t<_View>
    iter_move(const __iterator& __it) noexcept(noexcept(::cuda::std::ranges::iter_move(__it.__current_)))
    {
      return ::cuda::std::ranges::iter_move(__it.__current_);
    }

    _CCCL_TEMPLATE(class _View2 = _View)
    _CCCL_REQUIRES(indirectly_swappable<iterator_t<_View2>>)
    _CCCL_API friend constexpr void
    iter_swap(const __iterator& __x, const __iterator& __y) noexcept(__noexcept_swappable<iterator_t<_View2>>)
    {
      ::cuda::std::ranges::iter_swap(__x.__current_, __y.__current_);
    }
  };

  class __sentinel
  {
  public:
    sentinel_t<_View> __end_{};

    _CCCL_HIDE_FROM_ABI constexpr __sentinel() = default;

    _CCCL_API constexpr explicit __sentinel(filter_view& __parent)
        : __end_{::cuda::std::ranges::end(__parent.__base_)}
    {}

    [[nodiscard]] _CCCL_API constexpr sentinel_t<_View> base() const
      noexcept(is_nothrow_copy_constructible_v<sentinel_t<_View>>)
    {
      return __end_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __iterator& __x, const __sentinel& __y) noexcept(
      __is_cpp17_nothrow_equality_comparable_v<iterator_t<_View>, sentinel_t<_View>>)
    {
      return __x.__current_ == __y.__end_;
    }

#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __sentinel& __x, const __iterator& __y) noexcept(
      __is_cpp17_nothrow_equality_comparable_v<iterator_t<_View>, sentinel_t<_View>>)
    {
      return __y == __x;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __iterator& __x, const __sentinel& __y) noexcept(
      __is_cpp17_nothrow_equality_comparable_v<iterator_t<_View>, sentinel_t<_View>>)
    {
      return !(__x == __y);
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __sentinel& __x, const __iterator& __y) noexcept(
      __is_cpp17_nothrow_equality_comparable_v<iterator_t<_View>, sentinel_t<_View>>)
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017
  };

  _CCCL_TEMPLATE(class _View2 = _View, class _Pred2 = _Pred)
  _CCCL_REQUIRES(default_initializable<_View2> _CCCL_AND default_initializable<_Pred2>)
  _CCCL_API constexpr filter_view() noexcept(is_nothrow_default_constructible_v<_View2>
                                             && is_nothrow_default_constructible_v<_Pred2>)
      : view_interface<filter_view<_View, _Pred>>{}
      , __base_()
      , __pred_()
      , __cached_begin_()
  {}

  _CCCL_API constexpr explicit filter_view(_View __base, _Pred __pred) noexcept(
    is_nothrow_move_constructible_v<_View>
    && is_nothrow_constructible_v<__movable_box<_Pred>, in_place_t, add_rvalue_reference_t<_Pred>>)
      : __base_{::cuda::std::move(__base)}
      , __pred_{in_place, ::cuda::std::move(__pred)}
  {}

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(copy_constructible<_View2>)
  [[nodiscard]] _CCCL_API constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View2>)
  {
    return __base_;
  }

  [[nodiscard]] _CCCL_API constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>)
  {
    return ::cuda::std::move(__base_);
  }

  [[nodiscard]] _CCCL_API constexpr const _Pred& pred() const noexcept
  {
    return *__pred_;
  }

  [[nodiscard]] _CCCL_API constexpr __iterator begin()
  {
    _CCCL_ASSERT(__pred_.__has_value(),
                 "filter_view needs to have a valid predicate before calling begin() -- did a previous "
                 "assignment to this filter_view fail?");

    if constexpr (__use_cache)
    {
      if (!__cached_begin_.__has_value())
      {
        __cached_begin_.__emplace(::cuda::std::ranges::find_if(__base_, ::cuda::std::ref(*__pred_)));
      }
      return {*this, *__cached_begin_};
    }
    else
    {
      return {*this, ::cuda::std::ranges::find_if(__base_, ::cuda::std::ref(*__pred_))};
    }
  }

  [[nodiscard]] _CCCL_API constexpr auto end()
  {
    if constexpr (common_range<_View>)
    {
      return __iterator{*this, ::cuda::std::ranges::end(__base_)};
    }
    else
    {
      return __sentinel{*this};
    }
  }
};

template <class _Range, class _Pred>
_CCCL_HOST_DEVICE filter_view(_Range&&, _Pred) -> filter_view<ranges::views::all_t<_Range>, _Pred>;

_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(filter_view)

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__filter)

struct __fn
{
  template <class _Range, class _Pred>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Pred&& __pred) const noexcept(noexcept(
    ::cuda::std::ranges::filter_view{::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Pred>(__pred)}))
    -> decltype(::cuda::std::ranges::filter_view{
      ::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Pred>(__pred)})
  {
    return ::cuda::std::ranges::filter_view{::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Pred>(__pred)};
  }

  _CCCL_TEMPLATE(class _Pred)
  _CCCL_REQUIRES(constructible_from<decay_t<_Pred>, _Pred>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Pred&& __pred) const
    noexcept(is_nothrow_constructible_v<decay_t<_Pred>, _Pred>)
  {
    return ::cuda::std::ranges::__pipeable{::cuda::std::__bind_back(*this, ::cuda::std::forward<_Pred>(__pred))};
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto filter = __filter::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_FILTER_VIEW_H
