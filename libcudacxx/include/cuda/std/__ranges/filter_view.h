// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_FILTER_VIEW_H
#define _LIBCUDACXX___RANGES_FILTER_VIEW_H

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
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <class _View, class = void>
struct __filter_iterator_category
{};

// clang-format off
template<class _View>
struct __filter_iterator_category<_View, enable_if_t<forward_range<_View>>> {
  using _Cat = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using iterator_category =
    _If<derived_from<_Cat, bidirectional_iterator_tag>, bidirectional_iterator_tag,
    _If<derived_from<_Cat, forward_iterator_tag>,       forward_iterator_tag,
    /* else */                                          _Cat
  >>;
};
// clang-format on

#  if _CCCL_STD_VER >= 2020
template <input_range _View, indirect_unary_predicate<iterator_t<_View>> _Pred>
  requires view<_View> && is_object_v<_Pred>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _View,
          class _Pred,
          class = enable_if_t<view<_View>>,
          class = enable_if_t<input_range<_View>>,
          class = enable_if_t<is_object_v<_Pred>>,
          class = enable_if_t<indirect_unary_predicate<_Pred, iterator_t<_View>>>>
#  endif // _CCCL_STD_VER <= 2017
class filter_view : public view_interface<filter_view<_View, _Pred>>
{
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Pred> __pred_;
  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();

  // We cache the result of begin() to allow providing an amortized O(1) begin() whenever
  // the underlying range is at least a forward_range.
  static constexpr bool _UseCache = forward_range<_View>;
  using _Cache                    = _If<_UseCache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;
  _CCCL_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();

public:
  class __iterator : public __filter_iterator_category<_View>
  {
  public:
    _CCCL_NO_UNIQUE_ADDRESS iterator_t<_View> __current_ = iterator_t<_View>();
    _CCCL_NO_UNIQUE_ADDRESS filter_view* __parent_       = nullptr;

    using iterator_concept =
      _If<bidirectional_range<_View>,
          bidirectional_iterator_tag,
          _If<forward_range<_View>, forward_iterator_tag, /* else */ input_iterator_tag>>;
    // using iterator_category = inherited;
    using value_type      = range_value_t<_View>;
    using difference_type = range_difference_t<_View>;

// It seems nvcc has a bug where the noexcept specification is incorrectly set
#  if _CCCL_STD_VER >= 2020
    _CCCL_HIDE_FROM_ABI __iterator()
      requires default_initializable<iterator_t<_View>>
    = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(default_initializable<iterator_t<_View2>>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator() noexcept(is_nothrow_default_constructible_v<iterator_t<_View2>>) {}
#  endif // _CCCL_STD_VER <= 2017

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator(filter_view& __parent, iterator_t<_View> __current)
        : __current_(_CUDA_VSTD::move(__current))
        , __parent_(_CUDA_VSTD::addressof(__parent))
    {}

    _LIBCUDACXX_HIDE_FROM_ABI constexpr iterator_t<_View> const& base() const& noexcept
    {
      return __current_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI constexpr iterator_t<_View> base() &&
    {
      return _CUDA_VSTD::move(__current_);
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr range_reference_t<_View> operator*() const
    {
      return *__current_;
    }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(__has_arrow<iterator_t<_View2>>&& copyable<iterator_t<_View2>>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr iterator_t<_View> operator->() const
    {
      return __current_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator++()
    {
      __current_ = _CUDA_VRANGES::find_if(
        _CUDA_VSTD::move(++__current_), _CUDA_VRANGES::end(__parent_->__base_), _CUDA_VSTD::ref(*__parent_->__pred_));
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES((!forward_range<_View2>) )
    _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator++(int)
    {
      ++*this;
    }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(forward_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator operator++(int)
    {
      auto __tmp = *this;
      ++*this;
      return __tmp;
    }

    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(bidirectional_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator--()
    {
      do
      {
        --__current_;
      } while (!_CUDA_VSTD::invoke(*__parent_->__pred_, *__current_));
      return *this;
    }
    _LIBCUDACXX_TEMPLATE(class _View2 = _View)
    _LIBCUDACXX_REQUIRES(bidirectional_range<_View2>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator operator--(int)
    {
      auto tmp = *this;
      --*this;
      return tmp;
    }

    template <class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(__iterator const& __x, __iterator const& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(equality_comparable<iterator_t<_View2>>)
    {
      return __x.__current_ == __y.__current_;
    }
#  if _CCCL_STD_VER <= 2017
    template <class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(__iterator const& __x, __iterator const& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(equality_comparable<iterator_t<_View2>>)
    {
      return __x.__current_ != __y.__current_;
    }
#  endif // _CCCL_STD_VER <= 2017

    // MSVC falls over its feet if this is not a template
    template <class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr range_rvalue_reference_t<_View2>
    iter_move(__iterator const& __it) noexcept(noexcept(_CUDA_VRANGES::iter_move(__it.__current_)))
    {
      return _CUDA_VRANGES::iter_move(__it.__current_);
    }

    template <class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
    iter_swap(__iterator const& __x, __iterator const& __y) noexcept(__noexcept_swappable<iterator_t<_View2>>)
      _LIBCUDACXX_TRAILING_REQUIRES(void)(indirectly_swappable<iterator_t<_View2>>)
    {
      return _CUDA_VRANGES::iter_swap(__x.__current_, __y.__current_);
    }
  };

  class __sentinel
  {
  public:
    sentinel_t<_View> __end_ = sentinel_t<_View>();

    _CCCL_HIDE_FROM_ABI __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __sentinel(filter_view& __parent)
        : __end_(_CUDA_VRANGES::end(__parent.__base_))
    {}

    _LIBCUDACXX_HIDE_FROM_ABI constexpr sentinel_t<_View> base() const
    {
      return __end_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator==(__iterator const& __x, __sentinel const& __y)
    {
      return __x.__current_ == __y.__end_;
    }
#  if _CCCL_STD_VER <= 2017
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator==(__sentinel const& __x, __iterator const& __y)
    {
      return __y.__current_ == __x.__end_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator!=(__iterator const& __x, __sentinel const& __y)
    {
      return __x.__current_ != __y.__end_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator!=(__sentinel const& __x, __iterator const& __y)
    {
      return __y.__current_ != __x.__end_;
    }
#  endif // _CCCL_STD_VER <= 2017
  };

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI filter_view()
    requires default_initializable<_View> && default_initializable<_Pred>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(default_initializable<_View2>&& default_initializable<_Pred>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr filter_view() noexcept(
    is_nothrow_default_constructible_v<_View2> && is_nothrow_default_constructible_v<_Pred>)
      : view_interface<filter_view<_View, _Pred>>()
  {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr filter_view(_View __base, _Pred __pred)
      : view_interface<filter_view<_View, _Pred>>()
      , __pred_(in_place, _CUDA_VSTD::move(__pred))
      , __base_(_CUDA_VSTD::move(__base))
  {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _View base() const&
  {
    return __base_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _View base() &&
  {
    return _CUDA_VSTD::move(__base_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Pred const& pred() const
  {
    return *__pred_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator begin()
  {
    _CCCL_ASSERT(__pred_.__has_value(),
                 "Trying to call begin() on a filter_view that does not have a valid predicate.");
    if constexpr (_UseCache)
    {
      if (!__cached_begin_.__has_value())
      {
        __cached_begin_.__emplace(_CUDA_VRANGES::find_if(__base_, _CUDA_VSTD::ref(*__pred_)));
      }
      return {*this, *__cached_begin_};
    }
    else
    {
      return {*this, _CUDA_VRANGES::find_if(__base_, _CUDA_VSTD::ref(*__pred_))};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end()
  {
    if constexpr (common_range<_View>)
    {
      return __iterator{*this, _CUDA_VRANGES::end(__base_)};
    }
    else
    {
      return __sentinel{*this};
    }
    _CCCL_UNREACHABLE();
  }
};

template <class _Range, class _Pred>
_CCCL_HOST_DEVICE filter_view(_Range&&, _Pred) -> filter_view<_CUDA_VIEWS::all_t<_Range>, _Pred>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__filter)
struct __fn
{
  template <class _Range, class _Pred>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, _Pred&& __pred) const
    noexcept(noexcept(/**/ filter_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred))))
      -> filter_view<all_t<_Range>, remove_cvref_t<_Pred>>
  {
    return /*-----------*/ filter_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Pred>(__pred));
  }

  _LIBCUDACXX_TEMPLATE(class _Pred)
  _LIBCUDACXX_REQUIRES(constructible_from<decay_t<_Pred>, _Pred>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Pred&& __pred) const
    noexcept(is_nothrow_constructible_v<decay_t<_Pred>, _Pred>)
  {
    return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Pred>(__pred)));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto filter = __filter::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___RANGES_FILTER_VIEW_H
