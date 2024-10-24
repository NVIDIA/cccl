// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_JOIN_VIEW_H
#define _LIBCUDACXX___RANGES_JOIN_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/defaultable_box.h>
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/as_lvalue.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <class>
struct __join_view_iterator_category
{};

template <class _View>
  requires is_reference_v<range_reference_t<_View>> && forward_range<_View> && forward_range<range_reference_t<_View>>
struct __join_view_iterator_category<_View>
{
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class, class = void, class = void, class = void>
struct __join_view_iterator_category
{};

template <class _View>
struct __join_view_iterator_category<_View,
                                     enable_if_t<is_reference_v<range_reference_t<_View>>>,
                                     enable_if_t<forward_range<_View>>,
                                     enable_if_t<forward_range<range_reference_t<_View>>>>
{
#  endif // _CCCL_STD_VER <= 2017
  using _OuterC = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using _InnerC = typename iterator_traits<iterator_t<range_reference_t<_View>>>::iterator_category;

  using iterator_category =
    _If<derived_from<_OuterC, bidirectional_iterator_tag> && derived_from<_InnerC, bidirectional_iterator_tag>
          && common_range<range_reference_t<_View>>,
        bidirectional_iterator_tag,
        _If<derived_from<_OuterC, forward_iterator_tag> && derived_from<_InnerC, forward_iterator_tag>,
            forward_iterator_tag,
            input_iterator_tag>>;
};

#  if _CCCL_STD_VER >= 2020
template <input_range _View>
  requires view<_View> && input_range<range_reference_t<_View>>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _View,
          enable_if_t<view<_View>, int>                           = 0,
          enable_if_t<input_range<_View>, int>                    = 0,
          enable_if_t<input_range<range_reference_t<_View>>, int> = 0>
#  endif // _CCCL_STD_VER <= 2017
class join_view : public view_interface<join_view<_View>>
{
private:
  using _InnerRange = range_reference_t<_View>;

  static constexpr bool _UseOuterCache = !forward_range<_View>;
  using _OuterCache                    = _If<_UseOuterCache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;

  static constexpr bool _UseInnerCache = !is_reference_v<_InnerRange>;
  using _InnerCache = _If<_UseInnerCache, __non_propagating_cache<remove_cvref_t<_InnerRange>>, __empty_cache>;

  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();
  _CCCL_NO_UNIQUE_ADDRESS _OuterCache __outer_;
  _CCCL_NO_UNIQUE_ADDRESS _InnerCache __inner_;

public:
  template <bool _Const>
  struct __iterator : public __join_view_iterator_category<__maybe_const<_Const, _View>>
  {
    template <bool>
    friend struct __iterator;

    friend join_view;

  private:
    using _Parent     = __maybe_const<_Const, join_view>;
    using _Base       = __maybe_const<_Const, _View>;
    using _Outer      = iterator_t<_Base>;
    using _Inner      = iterator_t<range_reference_t<_Base>>;
    using _InnerRange = range_reference_t<_View>;

    static_assert(!_Const || forward_range<_Base>, "Const can only be true when Base models forward_range.");

    static constexpr bool __ref_is_glvalue = is_reference_v<range_reference_t<_Base>>;

    static constexpr bool _OuterPresent         = forward_range<_Base>;
    using _OuterType                            = _If<_OuterPresent, _Outer, _CUDA_VRANGES::__empty_cache>;
    _CCCL_NO_UNIQUE_ADDRESS _OuterType __outer_ = _OuterType();
    __defaultable_box<_Inner> __inner_{};
    _Parent* __parent_ = nullptr;

    struct __get_outer_emplace
    {
      __iterator& __iter_;

      _LIBCUDACXX_HIDE_FROM_ABI constexpr __get_outer_emplace(__iterator& __iter) noexcept
          : __iter_(__iter)
      {}

      _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator()() const noexcept
      {
        return *__iter_.__get_outer();
      }
    };

    _LIBCUDACXX_HIDE_FROM_ABI constexpr auto&& __update_inner()
    {
      if constexpr (__ref_is_glvalue)
      {
        return *__get_outer();
      }
      else
      {
        return __parent_->__inner_.__emplace_from(__get_outer_emplace{*this});
      }
      _CCCL_UNREACHABLE();
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr void __satisfy()
    {
      for (; __get_outer() != _CUDA_VRANGES::end(__parent_->__base_); ++__get_outer())
      {
        auto&& __inner = __update_inner();
        __inner_       = _CUDA_VRANGES::begin(__inner);
        if (*__inner_ != _CUDA_VRANGES::end(__inner))
        {
          return;
        }
      }

      if constexpr (__ref_is_glvalue)
      {
        __inner_.__reset();
      }
    }

  public:
    _LIBCUDACXX_HIDE_FROM_ABI constexpr _Outer& __get_outer() noexcept
    {
      if constexpr (forward_range<_Base>)
      {
        return __outer_;
      }
      else
      {
        return *__parent_->__outer_;
      }
      _CCCL_UNREACHABLE();
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Outer& __get_outer() const noexcept
    {
      if constexpr (forward_range<_Base>)
      {
        return __outer_;
      }
      else
      {
        return *__parent_->__outer_;
      }
      _CCCL_UNREACHABLE();
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
    _LIBCUDACXX_REQUIRES((!forward_range<_Base2>) )
    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __iterator(_Parent& __parent)
        : __parent_(_CUDA_VSTD::addressof(__parent))
    {
      __satisfy();
    }

    _LIBCUDACXX_TEMPLATE(class _Base2 = _Base)
    _LIBCUDACXX_REQUIRES(forward_range<_Base2>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator(_Parent& __parent, _Outer __outer)
        : __outer_(_CUDA_VSTD::move(__outer))
        , __parent_(_CUDA_VSTD::addressof(__parent))
    {
      __satisfy();
    }

  public:
    using iterator_concept =
      _If<__ref_is_glvalue && bidirectional_range<_Base> && bidirectional_range<range_reference_t<_Base>>
            && common_range<range_reference_t<_Base>>,
          bidirectional_iterator_tag,
          _If<__ref_is_glvalue && forward_range<_Base> && forward_range<range_reference_t<_Base>>,
              forward_iterator_tag,
              input_iterator_tag>>;

    using value_type = range_value_t<range_reference_t<_Base>>;

    using difference_type = common_type_t<range_difference_t<_Base>, range_difference_t<range_reference_t<_Base>>>;

    _CCCL_HIDE_FROM_ABI __iterator() = default;

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
    _LIBCUDACXX_REQUIRES(_OtherConst _LIBCUDACXX_AND convertible_to<iterator_t<_View>, _Outer> _LIBCUDACXX_AND
                           convertible_to<iterator_t<_InnerRange>, _Inner>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator(__iterator<!_OtherConst> __i)
        : __outer_(_CUDA_VSTD::move(__i.__outer_))
        , __inner_(_CUDA_VSTD::move(__i.__inner_))
        , __parent_(__i.__parent_)
    {}

    _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
    {
      return **__inner_;
    }

    _LIBCUDACXX_TEMPLATE(class _Inner2 = _Inner)
    _LIBCUDACXX_REQUIRES(__has_arrow<_Inner2> _LIBCUDACXX_AND copyable<_Inner2>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr _Inner2 operator->() const
    {
      return *__inner_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator++()
    {
      if constexpr (__ref_is_glvalue)
      {
        if (++*__inner_ == _CUDA_VRANGES::end(_CUDA_VSTD::__as_lvalue(*__get_outer())))
        {
          ++__get_outer();
          __satisfy();
        }
      }
      else
      {
        if (++*__inner_ == _CUDA_VRANGES::end(_CUDA_VSTD::__as_lvalue(*__parent_->__inner_)))
        {
          ++__get_outer();
          __satisfy();
        }
      }
      return *this;
    }

    _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator++(int)
    {
      if constexpr (__ref_is_glvalue && forward_range<_Base> && forward_range<range_reference_t<_Base>>)
      {
        auto __tmp = *this;
        ++*this;
        return __tmp;
      }
      else
      {
        ++*this;
      }
    }

    _LIBCUDACXX_TEMPLATE(bool __ref_is_glvalue2 = __ref_is_glvalue)
    _LIBCUDACXX_REQUIRES(
      __ref_is_glvalue2 _LIBCUDACXX_AND bidirectional_range<_Base> _LIBCUDACXX_AND
        bidirectional_range<range_reference_t<_Base>> _LIBCUDACXX_AND common_range<range_reference_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator& operator--()
    {
      if (__outer_ == _CUDA_VRANGES::end(__parent_->__base_))
      {
        __inner_ = _CUDA_VRANGES::end(_CUDA_VSTD::__as_lvalue(*--__outer_));
      }

      // Skip empty inner ranges when going backwards.
      while (*__inner_ == _CUDA_VRANGES::begin(_CUDA_VSTD::__as_lvalue(*__outer_)))
      {
        __inner_ = _CUDA_VRANGES::end(_CUDA_VSTD::__as_lvalue(*--__outer_));
      }

      --*__inner_;
      return *this;
    }

    _LIBCUDACXX_TEMPLATE(bool __ref_is_glvalue2 = __ref_is_glvalue)
    _LIBCUDACXX_REQUIRES(
      __ref_is_glvalue2 _LIBCUDACXX_AND bidirectional_range<_Base> _LIBCUDACXX_AND
        bidirectional_range<range_reference_t<_Base>> _LIBCUDACXX_AND common_range<range_reference_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __iterator operator--(int)
    {
      auto __tmp = *this;
      --*this;
      return __tmp;
    }

    template <bool __ref_is_glvalue2 = __ref_is_glvalue>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(__ref_is_glvalue2&& equality_comparable<iterator_t<_Base>>&&
                                            equality_comparable<iterator_t<range_reference_t<_Base>>>)
    {
      return __x.__outer_ == __y.__outer_ && __x.__inner_ == __y.__inner_;
    }
#  if _CCCL_STD_VER <= 2017
    template <bool __ref_is_glvalue2 = __ref_is_glvalue>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const __iterator& __x, const __iterator& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(__ref_is_glvalue2&& equality_comparable<iterator_t<_Base>>&&
                                            equality_comparable<iterator_t<range_reference_t<_Base>>>)
    {
      return __x.__outer_ != __y.__outer_ || __x.__inner_ != __y.__inner_;
    }
#  endif // _CCCL_STD_VER <= 2017

    // MSVC falls over its feet if this is not a template
    template <class _View2 = _View>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr decltype(auto)
    iter_move(const __iterator& __i) noexcept(noexcept(_CUDA_VRANGES::iter_move(*__i.__inner_)))
    {
      return _CUDA_VRANGES::iter_move(*__i.__inner_);
    }

    template <class _Inner2 = _Inner>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
    iter_swap(const __iterator& __x, const __iterator& __y) noexcept(__noexcept_swappable<_Inner2>)
      _LIBCUDACXX_TRAILING_REQUIRES(void)(indirectly_swappable<_Inner2>)
    {
      return _CUDA_VRANGES::iter_swap(*__x.__inner_, *__y.__inner_);
    }
  };

  template <bool _Const>
  struct __sentinel
  {
    template <bool>
    friend struct __sentinel;

  private:
    using _Parent            = __maybe_const<_Const, join_view>;
    using _Base              = __maybe_const<_Const, _View>;
    sentinel_t<_Base> __end_ = sentinel_t<_Base>();

  public:
    _CCCL_HIDE_FROM_ABI __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __sentinel(_Parent& __parent)
        : __end_(_CUDA_VRANGES::end(__parent.__base_))
    {}

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
    _LIBCUDACXX_REQUIRES(_OtherConst _LIBCUDACXX_AND convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __sentinel(__sentinel<!_OtherConst> __s)
        : __end_(_CUDA_VSTD::move(__s.__end_))
    {}

    template <bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __x.__get_outer() == __y.__end_;
    }
#  if _CCCL_STD_VER <= 2017
    template <bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __y.__get_outer() == __x.__end_;
    }
    template <bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __x.__get_outer() != __y.__end_;
    }
    template <bool _OtherConst>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __y.__get_outer() != __x.__end_;
    }
#  endif // _CCCL_STD_VER <= 2017
  };

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI join_view()
    requires default_initializable<_View>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(default_initializable<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr join_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<join_view<_View>>()
  {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit join_view(_View __base)
      : view_interface<join_view<_View>>()
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

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin()
  {
    if constexpr (forward_range<_View>)
    {
      constexpr bool __use_const = __simple_view<_View> && is_reference_v<range_reference_t<_View>>;
      return __iterator<__use_const>{*this, _CUDA_VRANGES::begin(__base_)};
    }
    else
    {
      __outer_.__emplace(_CUDA_VRANGES::begin(__base_));
      return __iterator<false>{*this};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(forward_range<const _View2> _LIBCUDACXX_AND is_reference_v<range_reference_t<const _View2>>
                         _LIBCUDACXX_AND input_range<range_reference_t<const _View2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin() const
  {
    return __iterator<true>{*this, _CUDA_VRANGES::begin(__base_)};
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end()
  {
    if constexpr (forward_range<_View> && is_reference_v<_InnerRange> && forward_range<_InnerRange>
                  && common_range<_View> && common_range<_InnerRange>)
    {
      return __iterator<__simple_view<_View>>{*this, _CUDA_VRANGES::end(__base_)};
    }
    else
    {
      return __sentinel<__simple_view<_View>>{*this};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(forward_range<const _View2> _LIBCUDACXX_AND is_reference_v<range_reference_t<const _View2>>
                         _LIBCUDACXX_AND input_range<range_reference_t<const _View2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end() const
  {
    using _ConstInnerRange = range_reference_t<const _View>;
    if constexpr (forward_range<_ConstInnerRange> && common_range<const _View> && common_range<_ConstInnerRange>)
    {
      return __iterator<true>{*this, _CUDA_VRANGES::end(__base_)};
    }
    else
    {
      return __sentinel<true>{*this};
    }
    _CCCL_UNREACHABLE();
  }
};

template <class _Range>
_CCCL_HOST_DEVICE explicit join_view(_Range&&) -> join_view<_CUDA_VIEWS::all_t<_Range>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__join_view)
struct __fn : __range_adaptor_closure<__fn>
{
  template <class _Range>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const noexcept(
    noexcept(/**/ join_view<all_t<_Range&&>>(_CUDA_VSTD::forward<_Range>(__range)))) -> join_view<all_t<_Range&&>>
  {
    return /*-----------*/ join_view<all_t<_Range&&>>(_CUDA_VSTD::forward<_Range>(__range));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto join = __join_view::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___RANGES_JOIN_VIEW_H
