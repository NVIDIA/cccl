// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_ZIP_TRANSFORM_VIEW_H
#define _CUDA_STD___RANGES_ZIP_TRANSFORM_VIEW_H

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
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/empty_view.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_referenceable.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

template <bool _Const, class _Fn, class _ViewsList, class = void>
struct __zip_transform_iterator_category_base
{};

template <bool _Const, class _Fn, class... _Views>
struct __zip_transform_iterator_category_base<_Const,
                                              _Fn,
                                              __type_list<_Views...>,
                                              enable_if_t<forward_range<__maybe_const<_Const, zip_view<_Views...>>>>>
{
private:
  template <class _View>
  using __tag _CCCL_NODEBUG = typename iterator_traits<iterator_t<__maybe_const<_Const, _View>>>::iterator_category;

  static _CCCL_CONSTEVAL auto __get_iterator_category()
  {
    if constexpr (!is_reference_v<
                    invoke_result_t<__maybe_const<_Const, _Fn>&, range_reference_t<__maybe_const<_Const, _Views>>...>>)
    {
      return ::cuda::std::input_iterator_tag{};
    }
    else if constexpr ((derived_from<__tag<_Views>, random_access_iterator_tag> && ...))
    {
      return ::cuda::std::random_access_iterator_tag{};
    }
    else if constexpr ((derived_from<__tag<_Views>, bidirectional_iterator_tag> && ...))
    {
      return ::cuda::std::bidirectional_iterator_tag{};
    }
    else if constexpr ((derived_from<__tag<_Views>, forward_iterator_tag> && ...))
    {
      return ::cuda::std::forward_iterator_tag{};
    }
    else
    {
      return ::cuda::std::input_iterator_tag{};
    }
  }

public:
  using iterator_category = decltype(__get_iterator_category());
};

#if _CCCL_HAS_CONCEPTS()
template <move_constructible _Fn, input_range... _Views>
  requires(view<_Views> && ...) && (sizeof...(_Views) > 0)
       && is_object_v<_Fn> && regular_invocable<_Fn&, range_reference_t<_Views>...>
       && __can_reference<invoke_result_t<_Fn&, range_reference_t<_Views>...>>
#else
template <class _Fn,
          class... _Views,
          class = enable_if_t<move_constructible<_Fn>>,
          class = enable_if_t<(input_range<_Views> && ...)>,
          class = enable_if_t<is_object_v<_Fn>>,
          class = enable_if_t<regular_invocable<_Fn&, range_reference_t<_Views>...>>,
          class = enable_if_t<__can_reference<invoke_result_t<_Fn&, range_reference_t<_Views>...>>>>
#endif
class zip_transform_view : public view_interface<zip_transform_view<_Fn, _Views...>>
{
  zip_view<_Views...> __zip_;
  __movable_box<_Fn> __fun_;

  using _InnerView _CCCL_NODEBUG = zip_view<_Views...>;

  template <bool _Const>
  using __ziperator _CCCL_NODEBUG = iterator_t<__maybe_const<_Const, _InnerView>>;

  template <bool _Const>
  using __zentinel _CCCL_NODEBUG = sentinel_t<__maybe_const<_Const, _InnerView>>;

public:
  template <bool _Const>
  class __iterator
  {
    using _Parent _CCCL_NODEBUG = __maybe_const<_Const, zip_transform_view>;
    using _Base _CCCL_NODEBUG   = __maybe_const<_Const, _InnerView>;

    friend zip_transform_view<_Fn, _Views...>;

    _Parent* __parent_ = nullptr;
    __ziperator<_Const> __inner_;

    _CCCL_API constexpr __iterator(_Parent& __parent, __ziperator<_Const> __inner)
        : __parent_{::cuda::std::addressof(__parent)}
        , __inner_{::cuda::std::move(__inner)}
    {}

    _CCCL_API constexpr auto __get_deref_and_invoke() const noexcept
    {
      return [&__fun = *__parent_->__fun_](const auto&... __iters) noexcept(
               noexcept(::cuda::std::invoke(__fun, *__iters...))) -> decltype(auto) {
        return ::cuda::std::invoke(__fun, *__iters...);
      };
    }

  public:
    using iterator_concept = typename __ziperator<_Const>::iterator_concept;
    using value_type =
      remove_cvref_t<invoke_result_t<__maybe_const<_Const, _Fn>&, range_reference_t<__maybe_const<_Const, _Views>>...>>;
    using difference_type = range_difference_t<_Base>;

    _CCCL_HIDE_FROM_ABI __iterator() = default;

    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(_OtherConst _CCCL_AND convertible_to<__ziperator<false>, __ziperator<_OtherConst>>)
    _CCCL_API constexpr __iterator(__iterator<!_OtherConst> __i)
        : __parent_{__i.__parent_}
        , __inner_{::cuda::std::move(__i.__inner_)}
    {}

    [[nodiscard]] _CCCL_API constexpr decltype(auto) operator*() const
      noexcept(noexcept(::cuda::std::apply(__get_deref_and_invoke(), __inner_.__iterators())))
    {
      return ::cuda::std::apply(__get_deref_and_invoke(), __inner_.__iterators());
    }

    _CCCL_API constexpr __iterator& operator++()
    {
      ++__inner_;
      return *this;
    }

    _CCCL_API constexpr void operator++(int)
    {
      ++*this;
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(forward_range<_Base2>)
    _CCCL_API constexpr __iterator operator++(int)
    {
      auto __tmp = *this;

      ++*this;
      return __tmp;
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(bidirectional_range<_Base2>)
    _CCCL_API constexpr __iterator& operator--()
    {
      --__inner_;
      return *this;
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(bidirectional_range<_Base2>)
    _CCCL_API constexpr __iterator operator--(int)
    {
      auto __tmp = *this;

      --*this;
      return __tmp;
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API constexpr __iterator& operator+=(difference_type __x)
    {
      __inner_ += __x;
      return *this;
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API constexpr __iterator& operator-=(difference_type __x)
    {
      __inner_ -= __x;
      return *this;
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API constexpr decltype(auto) operator[](difference_type __n) const
    {
      return ::cuda::std::apply(
        [&](const auto&... __iters) -> decltype(auto) {
          return ::cuda::std::invoke(
            *__parent_->__fun_, __iters[iter_difference_t<::cuda::std::remove_cvref_t<decltype(__iters)>>(__n)]...);
        },
        __inner_.__iterators());
    }

    _CCCL_TEMPLATE(class _Const2 = _Const)
    _CCCL_REQUIRES(equality_comparable<__ziperator<_Const2>>)
    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    {
      return __x.__inner_ == __y.__inner_;
    }

#if _CCCL_STD_VER <= 2017
    _CCCL_TEMPLATE(class _Const2 = _Const)
    _CCCL_REQUIRES(equality_comparable<__ziperator<_Const2>>)
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __iterator& __x, const __iterator& __y)
    {
      return !(__x == __y);
    }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    {
      return __x.__inner_ <=> __y.__inner_;
    }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API friend constexpr __iterator operator+(const __iterator& __i, difference_type __n)
    {
      return __iterator{*__i.__parent_, __i.__inner_ + __n};
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API friend constexpr __iterator operator+(difference_type __n, const __iterator& __i)
    {
      return __iterator{*__i.__parent_, __i.__inner_ + __n};
    }

    _CCCL_TEMPLATE(class _Base2 = _Base)
    _CCCL_REQUIRES(random_access_range<_Base2>)
    _CCCL_API friend constexpr __iterator operator-(const __iterator& __i, difference_type __n)
    {
      return __iterator{*__i.__parent_, __i.__inner_ - __n};
    }

    _CCCL_TEMPLATE(class _Const2 = _Const)
    _CCCL_REQUIRES(sized_sentinel_for<__ziperator<_Const2>, __ziperator<_Const2>>)
    _CCCL_API friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    {
      return __x.__inner_ - __y.__inner_;
    }
  };

  template <bool _Const>
  class __sentinel
  {
    __zentinel<_Const> __inner_;

    friend zip_transform_view<_Fn, _Views...>;

    _CCCL_API constexpr explicit __sentinel(__zentinel<_Const> __inner)
        : __inner_{::cuda::std::move(__inner)}
    {}

  public:
    _CCCL_HIDE_FROM_ABI __sentinel() = default;

    _CCCL_TEMPLATE(bool _OtherConst = _Const)
    _CCCL_REQUIRES(_OtherConst _CCCL_AND convertible_to<__zentinel<false>, __zentinel<_Const>>)
    _CCCL_API constexpr __sentinel(__sentinel<!_OtherConst> __i)
        : __inner_{__i.__inner_}
    {}

    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>)
    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __iterator<_OtherConst>& __x, const __sentinel& __y)
    {
      return __x.__inner_ == __y.__inner_;
    }

#if _CCCL_STD_VER <= 2017
    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>)
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __iterator<_OtherConst>& __x, const __sentinel& __y)
    {
      return !(__x == __y);
    }

    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>)
    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __sentinel& __x, const __iterator<_OtherConst>& __y)
    {
      return __y == __x;
    }

    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>)
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __sentinel& __x, const __iterator<_OtherConst>& __y)
    {
      return __y != __x;
    }
#endif // _CCCL_STD_VER <= 2017

    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(sized_sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>)
    [[nodiscard]] _CCCL_API friend constexpr range_difference_t<__maybe_const<_OtherConst, _InnerView>>
    operator-(const __iterator<_OtherConst>& __x, const __sentinel& __y)
    {
      return __x.__inner_ - __y.__inner_;
    }

    _CCCL_TEMPLATE(bool _OtherConst)
    _CCCL_REQUIRES(sized_sentinel_for<__zentinel<_Const>, __ziperator<_OtherConst>>)
    [[nodiscard]] _CCCL_API friend constexpr range_difference_t<__maybe_const<_OtherConst, _InnerView>>
    operator-(const __sentinel& __x, const __iterator<_OtherConst>& __y)
    {
      return __x.__inner_ - __y.__inner_;
    }
  };

  _CCCL_HIDE_FROM_ABI zip_transform_view() = default;

  _CCCL_API constexpr explicit zip_transform_view(_Fn __fun, _Views... __views)
      : __zip_{::cuda::std::move(__views)...}
      , __fun_{::cuda::std::in_place, ::cuda::std::move(__fun)}
  {}

  [[nodiscard]] _CCCL_API constexpr auto begin()
  {
    return __iterator<false>{*this, __zip_.begin()};
  }

  _CCCL_TEMPLATE(class _InnerView2 = _InnerView, class _Fn2 = _Fn)
  _CCCL_REQUIRES(range<const _InnerView2> _CCCL_AND regular_invocable<const _Fn2&, range_reference_t<const _Views>...>)
  [[nodiscard]] _CCCL_API constexpr auto begin() const
  {
    return __iterator<true>{*this, __zip_.begin()};
  }

  [[nodiscard]] _CCCL_API constexpr auto end()
  {
    if constexpr (common_range<_InnerView>)
    {
      return __iterator<false>{*this, __zip_.end()};
    }
    else
    {
      return __sentinel<false>{__zip_.end()};
    }
  }

  _CCCL_TEMPLATE(class _InnerView2 = _InnerView, class _Fn2 = _Fn)
  _CCCL_REQUIRES(range<const _InnerView2> _CCCL_AND regular_invocable<const _Fn2&, range_reference_t<const _Views>...>)
  [[nodiscard]] _CCCL_API constexpr auto end() const
  {
    if constexpr (common_range<const _InnerView>)
    {
      return __iterator<true>{*this, __zip_.end()};
    }
    else
    {
      return __sentinel<true>{__zip_.end()};
    }
  }

  _CCCL_TEMPLATE(class _InnerView2 = _InnerView)
  _CCCL_REQUIRES(sized_range<_InnerView2>)
  [[nodiscard]] _CCCL_API constexpr auto size()
  {
    return __zip_.size();
  }

  _CCCL_TEMPLATE(class _InnerView2 = _InnerView)
  _CCCL_REQUIRES(sized_range<const _InnerView2>)
  [[nodiscard]] _CCCL_API constexpr auto size() const
  {
    return __zip_.size();
  }
};

template <class _Fn, class... _Ranges>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES zip_transform_view(_Fn, _Ranges&&...)
  -> zip_transform_view<_Fn, views::all_t<_Ranges>...>;

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__zip_transform)

struct __fn
{
  _CCCL_TEMPLATE(class _Fn)
  _CCCL_REQUIRES(move_constructible<decay_t<_Fn>> _CCCL_AND regular_invocable<decay_t<_Fn>&> _CCCL_AND
                   is_object_v<invoke_result_t<decay_t<_Fn>&>>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Fn&&) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(views::empty<decay_t<invoke_result_t<decay_t<_Fn>&>>>)))
  {
    return views::empty<decay_t<invoke_result_t<decay_t<_Fn>&>>>;
  }

  _CCCL_TEMPLATE(class _Fn, class... _Ranges)
  _CCCL_REQUIRES((sizeof...(_Ranges) > 0))
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Fn&& __fun, _Ranges&&... __rs) const
    noexcept(noexcept(zip_transform_view{::cuda::std::forward<_Fn>(__fun), ::cuda::std::forward<_Ranges>(__rs)...}))
      -> decltype(zip_transform_view{::cuda::std::forward<_Fn>(__fun), ::cuda::std::forward<_Ranges>(__rs)...})
  {
    return zip_transform_view{::cuda::std::forward<_Fn>(__fun), ::cuda::std::forward<_Ranges>(__rs)...};
  }
};

_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto zip_transform = __zip_transform::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_ZIP_TRANSFORM_VIEW_H
