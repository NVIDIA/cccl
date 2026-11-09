// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_JOIN_VIEW_H
#define _CUDA_STD___RANGES_JOIN_VIEW_H

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
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/optional_box.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/as_lvalue.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

_LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE

#if _CCCL_HAS_CONCEPTS()
template <class>
struct __join_view_iterator_category
{};

template <class _View>
  requires is_reference_v<range_reference_t<_View>> && forward_range<_View> && forward_range<range_reference_t<_View>>
struct __join_view_iterator_category<_View>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class, class = void, class = void, class = void>
struct __join_view_iterator_category
{};

template <class _View>
struct __join_view_iterator_category<_View,
                                     enable_if_t<is_reference_v<range_reference_t<_View>>>,
                                     enable_if_t<forward_range<_View>>,
                                     enable_if_t<forward_range<range_reference_t<_View>>>>
#endif // !_CCCL_HAS_CONCEPTS()
{
  using __outer_cat _CCCL_NODEBUG = typename iterator_traits<iterator_t<_View>>::iterator_category;
  using __inner_cat _CCCL_NODEBUG = typename iterator_traits<iterator_t<range_reference_t<_View>>>::iterator_category;

  using iterator_category = conditional_t<
    derived_from<__outer_cat, bidirectional_iterator_tag> && derived_from<__inner_cat, bidirectional_iterator_tag>
      && common_range<range_reference_t<_View>>,
    bidirectional_iterator_tag,
    conditional_t<derived_from<__outer_cat, forward_iterator_tag> && derived_from<__inner_cat, forward_iterator_tag>,
                  forward_iterator_tag,
                  input_iterator_tag>>;
};

_CCCL_DIAG_PUSH
// warning C4238: nonstandard extension used: class rvalue used as lvalue
//
// Fires from use of __as_lvalue() with /permissive- and /W4. There don't seem to be any
// source-level changes you can make to silence MSVC here so we must disable the warning.
_CCCL_DIAG_SUPPRESS_MSVC(4238)

#if _CCCL_HAS_CONCEPTS()
template <input_range _View>
  requires view<_View> && input_range<range_reference_t<_View>>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _View,
          enable_if_t<view<_View>, int>                           = 0,
          enable_if_t<input_range<_View>, int>                    = 0,
          enable_if_t<input_range<range_reference_t<_View>>, int> = 0>
#endif // !_CCCL_HAS_CONCEPTS()
class join_view : public view_interface<join_view<_View>>
{
  using __inner_range_type _CCCL_NODEBUG = range_reference_t<_View>;

  static constexpr bool __use_outer_cache = !forward_range<_View>;
  using __outer_cache_type _CCCL_NODEBUG =
    conditional_t<__use_outer_cache, __non_propagating_cache<iterator_t<_View>>, __empty_cache>;

  static constexpr bool __use_inner_cache = !is_reference_v<__inner_range_type>;
  using __inner_cache_type _CCCL_NODEBUG =
    conditional_t<__use_inner_cache, __non_propagating_cache<remove_cvref_t<__inner_range_type>>, __empty_cache>;

  // These should be [[no_unique_address]] but nvcc 12.9 - 13.1 tends to OOM and/or crash with
  // [[no_unique_address]] and reasonably complex views, so we can't use it.
  _View __base_{};
  __outer_cache_type __outer_{};
  __inner_cache_type __inner_{};

public:
  template <bool _Const>
  class __iterator : public __join_view_iterator_category<__maybe_const<_Const, _View>>
  {
  public:
    template <bool>
    friend class __iterator;

    friend join_view;

  private:
    using __parent _CCCL_NODEBUG      = __maybe_const<_Const, join_view>;
    using __base _CCCL_NODEBUG        = __maybe_const<_Const, _View>;
    using __outer _CCCL_NODEBUG       = iterator_t<__base>;
    using __inner _CCCL_NODEBUG       = iterator_t<range_reference_t<__base>>;
    using __inner_range _CCCL_NODEBUG = range_reference_t<_View>;

    static_assert(!_Const || forward_range<__base>, "Const can only be true when Base models forward_range.");

    static constexpr bool __ref_is_glvalue = is_reference_v<range_reference_t<__base>>;

    static constexpr bool __outer_present = forward_range<__base>;
    using __outer_type _CCCL_NODEBUG      = conditional_t<__outer_present, __outer, ::cuda::std::ranges::__empty_cache>;

    // __outer_ should be [[no_unique_address]] bute nvcc 12.9 - 13.1 tends to OOM and/or crash
    // with [[no_unique_address]] and reasonably complex views, so we can't use it.
    __outer_type __outer_{};
    __optional_box<__inner> __inner_{};
    __parent* __parent_ = nullptr;

    struct __get_outer_emplace
    {
      __iterator& __iter_;

      _CCCL_API constexpr decltype(auto) operator()() const noexcept
      {
        return *__iter_.__get_outer();
      }
    };

    _CCCL_API constexpr auto&& __update_inner()
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

    _CCCL_API constexpr void __satisfy()
    {
      for (; __get_outer() != ::cuda::std::ranges::__end_cpo{}(__parent_->__base_); ++__get_outer())
      {
        auto&& __inner_tmp = __update_inner();

        // Use the begin/end CPOs directly due to __tile__.
        __inner_ = ::cuda::std::ranges::__begin_cpo{}(__inner_tmp);
        if (*__inner_ != ::cuda::std::ranges::__end_cpo{}(__inner_tmp))
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
    [[nodiscard]] _CCCL_API constexpr __outer& __get_outer() noexcept
    {
      if constexpr (forward_range<__base>)
      {
        return __outer_;
      }
      else
      {
        return *__parent_->__outer_;
      }
      _CCCL_UNREACHABLE();
    }

    [[nodiscard]] _CCCL_API constexpr const __outer& __get_outer() const noexcept
    {
      if constexpr (forward_range<__base>)
      {
        return __outer_;
      }
      else
      {
        return *__parent_->__outer_;
      }
      _CCCL_UNREACHABLE();
    }

    _CCCL_TEMPLATE(class _Base2 = __base)
    _CCCL_REQUIRES((!forward_range<_Base2>) )
    _CCCL_API constexpr explicit __iterator(__parent& __parent)
        : __parent_{::cuda::std::addressof(__parent)}
    {
      __satisfy();
    }

    _CCCL_TEMPLATE(class _Base2 = __base)
    _CCCL_REQUIRES(forward_range<_Base2>)
    _CCCL_API constexpr __iterator(__parent& __parent, __outer __outer)
        : __outer_{::cuda::std::move(__outer)}
        , __parent_{::cuda::std::addressof(__parent)}
    {
      __satisfy();
    }

    using iterator_concept =
      conditional_t<__ref_is_glvalue && bidirectional_range<__base> && bidirectional_range<range_reference_t<__base>>
                      && common_range<range_reference_t<__base>>,
                    bidirectional_iterator_tag,
                    conditional_t<__ref_is_glvalue && forward_range<__base> && forward_range<range_reference_t<__base>>,
                                  forward_iterator_tag,
                                  input_iterator_tag>>;

    using value_type = range_value_t<range_reference_t<__base>>;

    using difference_type = common_type_t<range_difference_t<__base>, range_difference_t<range_reference_t<__base>>>;

    _CCCL_HIDE_FROM_ABI __iterator() = default;

    _CCCL_TEMPLATE(bool _OtherConst = _Const, class _Inner2 = __inner, class _Outer2 = __outer)
    _CCCL_REQUIRES(_OtherConst _CCCL_AND convertible_to<iterator_t<_View>, _Outer2> _CCCL_AND
                     convertible_to<iterator_t<__inner_range>, _Inner2>)
    _CCCL_API constexpr __iterator(__iterator<!_OtherConst> __i)
        : __outer_{::cuda::std::move(__i.__outer_)}
        , __inner_{::cuda::std::move(__i.__inner_)}
        , __parent_{__i.__parent_}
    {}

    [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
    {
      return **__inner_;
    }

    _CCCL_TEMPLATE(class _Inner2 = __inner)
    _CCCL_REQUIRES(__has_arrow<_Inner2> _CCCL_AND copyable<_Inner2>)
    [[nodiscard]] _CCCL_API constexpr _Inner2 operator->() const
    {
      return *__inner_;
    }

    _CCCL_API constexpr __iterator& operator++()
    {
      if constexpr (__ref_is_glvalue)
      {
        if (++*__inner_ == ::cuda::std::ranges::__end_cpo{}(::cuda::std::__as_lvalue(*__get_outer())))
        {
          ++__get_outer();
          __satisfy();
        }
      }
      else
      {
        if (++*__inner_ == ::cuda::std::ranges::__end_cpo{}(::cuda::std::__as_lvalue(*__parent_->__inner_)))
        {
          ++__get_outer();
          __satisfy();
        }
      }
      return *this;
    }

    _CCCL_API constexpr decltype(auto) operator++(int)
    {
      if constexpr (__ref_is_glvalue && forward_range<__base> && forward_range<range_reference_t<__base>>)
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

    _CCCL_TEMPLATE(class _Base2 = __base)
    _CCCL_REQUIRES(is_reference_v<range_reference_t<_Base2>> _CCCL_AND bidirectional_range<_Base2> _CCCL_AND
                     bidirectional_range<range_reference_t<_Base2>> _CCCL_AND common_range<range_reference_t<_Base2>>)
    _CCCL_API constexpr __iterator& operator--()
    {
      if (__outer_ == ::cuda::std::ranges::__end_cpo{}(__parent_->__base_))
      {
#if _CCCL_COMPILER(MSVC) // MSVC2019 miscompiles when passing --__outer_ directly
        auto __tmp = --__outer_;

        __inner_ = ::cuda::std::ranges::__end_cpo{}(::cuda::std::__as_lvalue(*__tmp));
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
        __inner_ = ::cuda::std::ranges::__end_cpo{}(::cuda::std::__as_lvalue(*--__outer_));
#endif // !_CCCL_COMPILER(MSVC)
      }

      // Skip empty inner ranges when going backwards.
      while (*__inner_ == ::cuda::std::ranges::__begin_cpo{}(::cuda::std::__as_lvalue(*__outer_)))
      {
#if _CCCL_COMPILER(MSVC) // MSVC2019 miscompiles when passing --__outer_ directly
        auto __tmp = --__outer_;

        __inner_ = ::cuda::std::ranges::__end_cpo{}(::cuda::std::__as_lvalue(*__tmp));
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
        __inner_ = ::cuda::std::ranges::__end_cpo{}(::cuda::std::__as_lvalue(*--__outer_));
#endif // !_CCCL_COMPILER(MSVC)
      }

      --*__inner_;
      return *this;
    }

    _CCCL_TEMPLATE(class _Base2 = __base)
    _CCCL_REQUIRES(is_reference_v<range_reference_t<_Base2>> _CCCL_AND bidirectional_range<_Base2> _CCCL_AND
                     bidirectional_range<range_reference_t<_Base2>> _CCCL_AND common_range<range_reference_t<_Base2>>)
    _CCCL_API constexpr __iterator operator--(int)
    {
      auto __tmp = *this;

      --*this;
      return __tmp;
    }

    template <bool __ref_is_glvalue2 = __ref_is_glvalue>
    _CCCL_API friend constexpr auto operator==(const __iterator& __x, const __iterator& __y)
      _CCCL_TRAILING_REQUIRES(bool)(__ref_is_glvalue2&& equality_comparable<iterator_t<__base>>&&
                                      equality_comparable<iterator_t<range_reference_t<__base>>>)
    {
      return __x.__outer_ == __y.__outer_ && __x.__inner_ == __y.__inner_;
    }
#if _CCCL_STD_VER <= 2017
    template <bool __ref_is_glvalue2 = __ref_is_glvalue>
    _CCCL_API friend constexpr auto operator!=(const __iterator& __x, const __iterator& __y)
      _CCCL_TRAILING_REQUIRES(bool)(__ref_is_glvalue2&& equality_comparable<iterator_t<__base>>&&
                                      equality_comparable<iterator_t<range_reference_t<__base>>>)
    {
      return __x.__outer_ != __y.__outer_ || __x.__inner_ != __y.__inner_;
    }
#endif // _CCCL_STD_VER <= 2017

    // MSVC falls over its feet if this is not a template
    template <class _View2 = _View>
    _CCCL_API friend constexpr decltype(auto)
    iter_move(const __iterator& __i) noexcept(noexcept(::cuda::std::ranges::__iter_move_cpo{}(*__i.__inner_)))
    {
      return ::cuda::std::ranges::__iter_move_cpo{}(*__i.__inner_);
    }

    _CCCL_TEMPLATE(class _Inner2 = __inner)
    _CCCL_REQUIRES(indirectly_swappable<_Inner2>)
    _CCCL_API friend constexpr void
    iter_swap(const __iterator& __x, const __iterator& __y) noexcept(__noexcept_swappable<_Inner2>)
    {
      ::cuda::std::ranges::__iter_swap_cpo{}(*__x.__inner_, *__y.__inner_);
    }
  };

  template <bool _Const>
  class __sentinel
  {
    using __parent _CCCL_NODEBUG = __maybe_const<_Const, join_view>;
    using __base _CCCL_NODEBUG   = __maybe_const<_Const, _View>;

    template <bool _OtherConst>
    using __base2 = __maybe_const<_OtherConst, _View>;

    sentinel_t<__base> __end_{};

  public:
    template <bool>
    friend class __sentinel;

    _CCCL_HIDE_FROM_ABI __sentinel() = default;

    _CCCL_API constexpr explicit __sentinel(__parent& __parent)
        : __end_{::cuda::std::ranges::__end_cpo{}(__parent.__base_)}
    {}

    _CCCL_TEMPLATE(bool _OtherConst = _Const)
    _CCCL_REQUIRES(_OtherConst _CCCL_AND convertible_to<sentinel_t<_View>, sentinel_t<__base2<_OtherConst>>>)
    _CCCL_API constexpr __sentinel(__sentinel<!_OtherConst> __s)
        : __end_{::cuda::std::move(__s.__end_)}
    {}

    template <bool _OtherConst>
    _CCCL_API friend constexpr auto operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<__base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __x.__get_outer() == __y.__end_;
    }

#if _CCCL_STD_VER <= 2017
    template <bool _OtherConst>
    _CCCL_API friend constexpr auto operator==(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<__base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __y.__get_outer() == __x.__end_;
    }

    template <bool _OtherConst>
    _CCCL_API friend constexpr auto operator!=(const __iterator<_OtherConst>& __x, const __sentinel& __y)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<__base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __x.__get_outer() != __y.__end_;
    }

    template <bool _OtherConst>
    _CCCL_API friend constexpr auto operator!=(const __sentinel& __x, const __iterator<_OtherConst>& __y)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<__base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __y.__get_outer() != __x.__end_;
    }
#endif // _CCCL_STD_VER <= 2017
  };

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI join_view()
    requires default_initializable<_View>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(default_initializable<_View2>)
  _CCCL_API constexpr join_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<join_view<_View>>{}
  {}
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr explicit join_view(_View __base)
      : view_interface<join_view<_View>>{}
      , __base_{::cuda::std::move(__base)}
  {}

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(copy_constructible<_View2>)
  [[nodiscard]] _CCCL_API constexpr _View base() const& noexcept
  {
    return __base_;
  }

  [[nodiscard]] _CCCL_API constexpr _View base() &&
  {
    return ::cuda::std::move(__base_);
  }

  [[nodiscard]] _CCCL_API constexpr auto begin()
  {
    if constexpr (forward_range<_View>)
    {
      constexpr bool __use_const = __simple_view<_View> && is_reference_v<range_reference_t<_View>>;

      return __iterator</*const*/ __use_const>{*this, ::cuda::std::ranges::__begin_cpo{}(__base_)};
    }
    else
    {
      __outer_.__emplace(::cuda::std::ranges::__begin_cpo{}(__base_));
      return __iterator</*const*/ false>{*this};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(forward_range<const _View2> _CCCL_AND is_reference_v<range_reference_t<const _View2>> _CCCL_AND
                   input_range<range_reference_t<const _View2>>)
  [[nodiscard]] _CCCL_API constexpr __iterator</*const*/ true> begin() const
  {
    return __iterator</*const*/ true>{*this, ::cuda::std::ranges::__begin_cpo{}(__base_)};
  }

  [[nodiscard]] _CCCL_API constexpr auto end()
  {
    if constexpr (forward_range<_View> && is_reference_v<__inner_range_type> && forward_range<__inner_range_type>
                  && common_range<_View> && common_range<__inner_range_type>)
    {
      return __iterator</*const*/ __simple_view<_View>>{*this, ::cuda::std::ranges::__end_cpo{}(__base_)};
    }
    else
    {
      return __sentinel</*const*/ __simple_view<_View>>{*this};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(forward_range<const _View2> _CCCL_AND is_reference_v<range_reference_t<const _View2>> _CCCL_AND
                   input_range<range_reference_t<const _View2>>)
  [[nodiscard]] _CCCL_API constexpr auto end() const
  {
    using __const_inner_range _CCCL_NODEBUG = range_reference_t<const _View>;

    if constexpr (forward_range<__const_inner_range> && common_range<const _View> && common_range<__const_inner_range>)
    {
      return __iterator</*const*/ true>{*this, ::cuda::std::ranges::__end_cpo{}(__base_)};
    }
    else
    {
      return __sentinel</*const*/ true>{*this};
    }
    _CCCL_UNREACHABLE();
  }
};

_CCCL_DIAG_POP

template <class _Range>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES explicit join_view(_Range&&) -> join_view<::cuda::std::ranges::views::all_t<_Range>>;

_LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(join_view)

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS

_CCCL_BEGIN_NAMESPACE_CPO(__join_view)

struct __fn : __range_adaptor_closure<__fn>
{
  template <class _Range>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(join_view<all_t<_Range&&>>{::cuda::std::forward<_Range>(__range)}))
      -> decltype(join_view<all_t<_Range&&>>{::cuda::std::forward<_Range>(__range)})
  {
    return join_view<all_t<_Range&&>>{::cuda::std::forward<_Range>(__range)};
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto join = __join_view::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_JOIN_VIEW_H
