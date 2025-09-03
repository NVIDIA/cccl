// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_TAKE_VIEW_H
#define _CUDA_STD___RANGES_TAKE_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/ranges_min.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__functional/bind_back.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__fwd/string_view.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/counted_iterator.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/empty_view.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/iota_view.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/repeat_view.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/subrange.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/forward_like.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_CCCL_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <view _View>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _View, class = enable_if_t<view<_View>>>
#endif // !_CCCL_HAS_CONCEPTS()
class take_view : public view_interface<take_view<_View>>
{
  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();
  range_difference_t<_View> __count_    = 0;

public:
  template <bool _Const>
  class __sentinel
  {
    using _Base _CCCL_NODEBUG_ALIAS = __maybe_const<_Const, _View>;
    template <bool _OtherConst>
    using _Iter _CCCL_NODEBUG_ALIAS                  = counted_iterator<iterator_t<__maybe_const<_OtherConst, _View>>>;
    _CCCL_NO_UNIQUE_ADDRESS sentinel_t<_Base> __end_ = sentinel_t<_Base>();

    template <bool>
    friend class __sentinel;

  public:
    _CCCL_HIDE_FROM_ABI __sentinel() = default;

    _CCCL_API constexpr explicit __sentinel(sentinel_t<_Base> __end)
        : __end_(::cuda::std::move(__end))
    {}

    _CCCL_TEMPLATE(bool _OtherConst = _Const)
    _CCCL_REQUIRES(_OtherConst _CCCL_AND convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
    _CCCL_API constexpr __sentinel(__sentinel<!_OtherConst> __s)
        : __end_(::cuda::std::move(__s.__end_))
    {}

    [[nodiscard]] _CCCL_API constexpr sentinel_t<_Base> base() const
    {
      return __end_;
    }

    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const _Iter<_Const>& __lhs, const __sentinel& __rhs)
    {
      return __lhs.count() == 0 || __lhs.base() == __rhs.__end_;
    }
#if _CCCL_STD_VER <= 2017
    [[nodiscard]] _CCCL_API friend constexpr bool operator==(const __sentinel& __lhs, const _Iter<_Const>& __rhs)
    {
      return __rhs.count() == 0 || __rhs.base() == __lhs.__end_;
    }
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const _Iter<_Const>& __lhs, const __sentinel& __rhs)
    {
      return !(__lhs == __rhs);
    }
    [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const __sentinel& __lhs, const _Iter<_Const>& __rhs)
    {
      return !(__lhs == __rhs);
    }
#endif // _CCCL_STD_VER <= 2017

    template <bool _OtherConst = !_Const>
    [[nodiscard]] _CCCL_API friend constexpr auto operator==(const _Iter<_OtherConst>& __lhs, const __sentinel& __rhs)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __lhs.count() == 0 || __lhs.base() == __rhs.__end_;
    }
#if _CCCL_STD_VER <= 2017
    template <bool _OtherConst = !_Const>
    [[nodiscard]] _CCCL_API friend constexpr auto operator==(const __sentinel& __lhs, const _Iter<_OtherConst>& __rhs)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __rhs.count() == 0 || __rhs.base() == __lhs.__end_;
    }
    template <bool _OtherConst = !_Const>
    [[nodiscard]] _CCCL_API friend constexpr auto operator!=(const _Iter<_OtherConst>& __lhs, const __sentinel& __rhs)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return !(__lhs == __rhs);
    }
    template <bool _OtherConst = !_Const>
    [[nodiscard]] _CCCL_API friend constexpr auto operator!=(const __sentinel& __lhs, const _Iter<_OtherConst>& __rhs)
      _CCCL_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return !(__lhs == __rhs);
    }
#endif // _CCCL_STD_VER <= 2017
  };

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI take_view()
    requires default_initializable<_View>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(default_initializable<_View2>)
  _CCCL_API constexpr take_view() noexcept(is_nothrow_default_constructible_v<_View2>) {}
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr take_view(_View __base, range_difference_t<_View> __count)
      : __base_(::cuda::std::move(__base))
      , __count_(__count)
  {}

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(copy_constructible<_View2>)
  [[nodiscard]] _CCCL_API constexpr _View base() const&
  {
    return __base_;
  }

  [[nodiscard]] _CCCL_API constexpr _View base() &&
  {
    return ::cuda::std::move(__base_);
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES((!__simple_view<_View2>) )
  [[nodiscard]] _CCCL_API constexpr auto begin()
  {
    if constexpr (sized_range<_View>)
    {
      if constexpr (random_access_range<_View>)
      {
        return ::cuda::std::ranges::begin(__base_);
      }
      else
      {
        using _DifferenceT = range_difference_t<_View>;
        auto __size        = size();
        return counted_iterator(::cuda::std::ranges::begin(__base_), static_cast<_DifferenceT>(__size));
      }
    }
    else
    {
      return counted_iterator(::cuda::std::ranges::begin(__base_), __count_);
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(range<const _View2>)
  [[nodiscard]] _CCCL_API constexpr auto begin() const
  {
    if constexpr (sized_range<const _View>)
    {
      if constexpr (random_access_range<const _View>)
      {
        return ::cuda::std::ranges::begin(__base_);
      }
      else
      {
        using _DifferenceT = range_difference_t<const _View>;
        auto __size        = size();
        return counted_iterator(::cuda::std::ranges::begin(__base_), static_cast<_DifferenceT>(__size));
      }
    }
    else
    {
      return counted_iterator(::cuda::std::ranges::begin(__base_), __count_);
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES((!__simple_view<_View2>) )
  [[nodiscard]] _CCCL_API constexpr auto end()
  {
    if constexpr (sized_range<_View>)
    {
      if constexpr (random_access_range<_View>)
      {
        return ::cuda::std::ranges::begin(__base_) + size();
      }
      else
      {
        return default_sentinel;
      }
    }
    else
    {
      return __sentinel<false>{::cuda::std::ranges::end(__base_)};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(range<const _View2>)
  [[nodiscard]] _CCCL_API constexpr auto end() const
  {
    if constexpr (sized_range<const _View>)
    {
      if constexpr (random_access_range<const _View>)
      {
        return ::cuda::std::ranges::begin(__base_) + size();
      }
      else
      {
        return default_sentinel;
      }
    }
    else
    {
      return __sentinel<true>{::cuda::std::ranges::end(__base_)};
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(sized_range<_View2>)
  [[nodiscard]] _CCCL_API constexpr auto size()
  {
    const auto __n = ::cuda::std::ranges::size(__base_);
    return (::cuda::std::ranges::min) (__n, static_cast<decltype(__n)>(__count_));
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(sized_range<const _View2>)
  [[nodiscard]] _CCCL_API constexpr auto size() const
  {
    auto __n = ::cuda::std::ranges::size(__base_);
    return (::cuda::std::ranges::min) (__n, static_cast<decltype(__n)>(__count_));
  }
};

template <class _Range>
_CCCL_HOST_DEVICE take_view(_Range&&, range_difference_t<_Range>)
  -> take_view<::cuda::std::ranges::views::all_t<_Range>>;

template <class _Tp>
inline constexpr bool enable_borrowed_range<take_view<_Tp>> = enable_borrowed_range<_Tp>;

_CCCL_END_NAMESPACE_RANGES

_CCCL_BEGIN_NAMESPACE_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__take)

template <class _Tp>
inline constexpr bool __is_empty_view = false;

template <class _Tp>
inline constexpr bool __is_empty_view<empty_view<_Tp>> = true;

template <class _Tp>
inline constexpr bool __is_passthrough_specialization = false;

template <class _Tp, size_t _Extent>
inline constexpr bool __is_passthrough_specialization<span<_Tp, _Extent>> = true;

template <class _CharT, class _Traits>
inline constexpr bool __is_passthrough_specialization<basic_string_view<_CharT, _Traits>> = true;

template <class _Iter, class _Sent, ::cuda::std::ranges::subrange_kind _Kind>
inline constexpr bool __is_passthrough_specialization<::cuda::std::ranges::subrange<_Iter, _Sent, _Kind>> = true;

template <class _Tp>
inline constexpr bool __is_iota_specialization = false;

template <class _Np, class _Bound>
inline constexpr bool __is_iota_specialization<iota_view<_Np, _Bound>> = true;

template <class _Tp, class = void>
struct __passthrough_type;

template <class _Tp, size_t _Extent>
struct __passthrough_type<span<_Tp, _Extent>>
{
  using type _CCCL_NODEBUG_ALIAS = span<_Tp>;
};

template <class _CharT, class _Traits>
struct __passthrough_type<basic_string_view<_CharT, _Traits>>
{
  using type = _CCCL_NODEBUG_ALIAS basic_string_view<_CharT, _Traits>;
};

template <class _Iter, class _Sent, ::cuda::std::ranges::subrange_kind _Kind>
struct __passthrough_type<::cuda::std::ranges::subrange<_Iter, _Sent, _Kind>,
                          void_t<typename ::cuda::std::ranges::subrange<_Iter>>>
{
  using type = _CCCL_NODEBUG_ALIAS ::cuda::std::ranges::subrange<_Iter>;
};

template <class _Tp>
using __passthrough_type_t _CCCL_NODEBUG_ALIAS = typename __passthrough_type<_Tp>::type;

template <class _Range, class _Np>
_CCCL_CONCEPT __use_empty = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(convertible_to<_Np, range_difference_t<_Range>>), requires(__is_empty_view<remove_cvref_t<_Range>>));

template <class _Range, class _Np>
_CCCL_CONCEPT __use_passthrough = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(convertible_to<_Np, range_difference_t<_Range>>),
  requires(!__is_empty_view<remove_cvref_t<_Range>>),
  requires(random_access_range<remove_cvref_t<_Range>>),
  requires(sized_range<remove_cvref_t<_Range>>),
  requires(__is_passthrough_specialization<remove_cvref_t<_Range>>));

template <class _Range, class _Np>
_CCCL_CONCEPT __use_iota = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(convertible_to<_Np, range_difference_t<_Range>>),
  requires(!__is_empty_view<remove_cvref_t<_Range>>),
  requires(random_access_range<remove_cvref_t<_Range>>),
  requires(sized_range<remove_cvref_t<_Range>>),
  requires(__is_iota_specialization<remove_cvref_t<_Range>>));

template <class _Range, class _Np>
_CCCL_CONCEPT __use_generic = _CCCL_REQUIRES_EXPR((_Range, _Np))(
  requires(!__use_empty<_Range, _Np>),
  requires(!__is_repeat_specialization<remove_cvref_t<_Range>>),
  requires(!__use_passthrough<_Range, _Np>),
  requires(!__use_iota<_Range, _Np>));

struct __fn
{
  // [range.take.overview]: the `empty_view` case.
  _CCCL_TEMPLATE(class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>)
  _CCCL_REQUIRES(__use_empty<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&&) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(::cuda::std::forward<_Range>(__range)))) -> _RawRange
  {
    return _LIBCUDACXX_AUTO_CAST(::cuda::std::forward<_Range>(__range));
  }

  // [range.take.overview]: the `span | basic_string_view | subrange` case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(__use_passthrough<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __rng, _Np&& __n) const
    noexcept(noexcept(__passthrough_type_t<_RawRange>(
      ::cuda::std::ranges::begin(__rng),
      ::cuda::std::ranges::begin(__rng)
        + ::cuda::std::min<_Dist>(::cuda::std::ranges::distance(__rng), ::cuda::std::forward<_Np>(__n)))))
      -> __passthrough_type_t<_RawRange>
  {
    return __passthrough_type_t<_RawRange>(
      ::cuda::std::ranges::begin(__rng),
      ::cuda::std::ranges::begin(__rng)
        + ::cuda::std::min<_Dist>(::cuda::std::ranges::distance(__rng), ::cuda::std::forward<_Np>(__n)));
  }

  // [range.take.overview]: the `repeat_view` "_RawRange models sized_range" case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(convertible_to<_Np, range_difference_t<_Range>> _CCCL_AND
                   __is_repeat_specialization<_RawRange> _CCCL_AND sized_range<_RawRange>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&& __n) const noexcept(noexcept(views::repeat(
    ::cuda::std::forward_like<_Range>(*__range.__value_),
    ::cuda::std::min<_Dist>(ranges::distance(__range), ::cuda::std::forward<_Np>(__n))))) -> _RawRange
  {
    return views::repeat(::cuda::std::forward_like<_Range>(*__range.__value_),
                         ::cuda::std::min<_Dist>(ranges::distance(__range), ::cuda::std::forward<_Np>(__n)));
  }

  // [range.take.overview]: the `repeat_view` "otherwise" case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(convertible_to<_Np, range_difference_t<_Range>> _CCCL_AND
                   __is_repeat_specialization<_RawRange> _CCCL_AND(!sized_range<_RawRange>))
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&& __n) const
    noexcept(noexcept(views::repeat(::cuda::std::forward_like<_Range>(*__range.__value_), static_cast<_Dist>(__n))))
      -> repeat_view<range_value_t<_RawRange>, _Dist>
  {
    return views::repeat(::cuda::std::forward_like<_Range>(*__range.__value_), static_cast<_Dist>(__n));
  }

  // [range.take.overview]: the `iota_view` case.
  _CCCL_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _CCCL_REQUIRES(__use_iota<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __rng, _Np&& __n) const
    noexcept(noexcept(::cuda::std::ranges::iota_view(
      *::cuda::std::ranges::begin(__rng),
      *::cuda::std::ranges::begin(__rng)
        + ::cuda::std::min<_Dist>(::cuda::std::ranges::distance(__rng), ::cuda::std::forward<_Np>(__n)))))
      -> iota_view<range_value_t<_RawRange>, _Dist>
  {
    return ::cuda::std::ranges::iota_view(
      *::cuda::std::ranges::begin(__rng),
      *::cuda::std::ranges::begin(__rng)
        + ::cuda::std::min<_Dist>(::cuda::std::ranges::distance(__rng), ::cuda::std::forward<_Np>(__n)));
  }

  // [range.take.overview]: the "otherwise" case.
  _CCCL_TEMPLATE(class _Range, class _Np)
  _CCCL_REQUIRES(__use_generic<_Range, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range, _Np&& __n) const
    noexcept(noexcept(take_view(::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Np>(__n))))
      -> decltype(take_view(::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Np>(__n)))
  {
    return take_view(::cuda::std::forward<_Range>(__range), ::cuda::std::forward<_Np>(__n));
  }

  _CCCL_TEMPLATE(class _Np)
  _CCCL_REQUIRES(constructible_from<decay_t<_Np>, _Np>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Np&& __n) const
    noexcept(is_nothrow_constructible_v<decay_t<_Np>, _Np>)
  {
    return __pipeable(::cuda::std::__bind_back(*this, ::cuda::std::forward<_Np>(__n)));
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto take = __take::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_TAKE_VIEW_H
