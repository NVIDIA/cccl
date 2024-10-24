// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_TAKE_VIEW_H
#define _LIBCUDACXX___RANGES_TAKE_VIEW_H

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
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  include <cuda/std/__fwd/string_view.h>
#endif // _LIBCUDACXX_HAS_STRING_VIEW
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
#include <cuda/std/__utility/move.h>

_CCCL_PUSH_MACROS

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <view _View>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _View, class = enable_if_t<view<_View>>>
#  endif // _CCCL_STD_VER <= 2017
class take_view : public view_interface<take_view<_View>>
{
  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();
  range_difference_t<_View> __count_    = 0;

public:
  template <bool _Const>
  class __sentinel
  {
    using _Base = __maybe_const<_Const, _View>;
    template <bool _OtherConst>
    using _Iter                                      = counted_iterator<iterator_t<__maybe_const<_OtherConst, _View>>>;
    _CCCL_NO_UNIQUE_ADDRESS sentinel_t<_Base> __end_ = sentinel_t<_Base>();

    template <bool>
    friend class __sentinel;

  public:
    _CCCL_HIDE_FROM_ABI __sentinel() = default;

    _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __sentinel(sentinel_t<_Base> __end)
        : __end_(_CUDA_VSTD::move(__end))
    {}

    _LIBCUDACXX_TEMPLATE(bool _OtherConst = _Const)
    _LIBCUDACXX_REQUIRES(_OtherConst&& convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
    _LIBCUDACXX_HIDE_FROM_ABI constexpr __sentinel(__sentinel<!_OtherConst> __s)
        : __end_(_CUDA_VSTD::move(__s.__end_))
    {}

    _LIBCUDACXX_HIDE_FROM_ABI constexpr sentinel_t<_Base> base() const
    {
      return __end_;
    }

    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator==(const _Iter<_Const>& __lhs, const __sentinel& __rhs)
    {
      return __lhs.count() == 0 || __lhs.base() == __rhs.__end_;
    }
#  if _CCCL_STD_VER <= 2017
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator==(const __sentinel& __lhs, const _Iter<_Const>& __rhs)
    {
      return __rhs.count() == 0 || __rhs.base() == __lhs.__end_;
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator!=(const _Iter<_Const>& __lhs, const __sentinel& __rhs)
    {
      return !(__lhs == __rhs);
    }
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool operator!=(const __sentinel& __lhs, const _Iter<_Const>& __rhs)
    {
      return !(__lhs == __rhs);
    }
#  endif // _CCCL_STD_VER <= 2017

    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const _Iter<_OtherConst>& __lhs, const __sentinel& __rhs)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __lhs.count() == 0 || __lhs.base() == __rhs.__end_;
    }
#  if _CCCL_STD_VER <= 2017
    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator==(const __sentinel& __lhs, const _Iter<_OtherConst>& __rhs)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return __rhs.count() == 0 || __rhs.base() == __lhs.__end_;
    }
    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const _Iter<_OtherConst>& __lhs, const __sentinel& __rhs)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return !(__lhs == __rhs);
    }
    template <bool _OtherConst = !_Const>
    _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto operator!=(const __sentinel& __lhs, const _Iter<_OtherConst>& __rhs)
      _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
    {
      return !(__lhs == __rhs);
    }
#  endif // _CCCL_STD_VER <= 2017
  };

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI take_view()
    requires default_initializable<_View>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(default_initializable<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr take_view() noexcept(is_nothrow_default_constructible_v<_View2>) {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr take_view(_View __base, range_difference_t<_View> __count)
      : __base_(_CUDA_VSTD::move(__base))
      , __count_(__count)
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

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES((!__simple_view<_View2>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin()
  {
    if constexpr (sized_range<_View>)
    {
      if constexpr (random_access_range<_View>)
      {
        return _CUDA_VRANGES::begin(__base_);
      }
      else
      {
        using _DifferenceT = range_difference_t<_View>;
        auto __size        = size();
        return counted_iterator(_CUDA_VRANGES::begin(__base_), static_cast<_DifferenceT>(__size));
      }
    }
    else
    {
      return counted_iterator(_CUDA_VRANGES::begin(__base_), __count_);
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin() const
  {
    if constexpr (sized_range<const _View>)
    {
      if constexpr (random_access_range<const _View>)
      {
        return _CUDA_VRANGES::begin(__base_);
      }
      else
      {
        using _DifferenceT = range_difference_t<const _View>;
        auto __size        = size();
        return counted_iterator(_CUDA_VRANGES::begin(__base_), static_cast<_DifferenceT>(__size));
      }
    }
    else
    {
      return counted_iterator(_CUDA_VRANGES::begin(__base_), __count_);
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES((!__simple_view<_View2>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end()
  {
    if constexpr (sized_range<_View>)
    {
      if constexpr (random_access_range<_View>)
      {
        return _CUDA_VRANGES::begin(__base_) + size();
      }
      else
      {
        return default_sentinel;
      }
    }
    else
    {
      return __sentinel<false>{_CUDA_VRANGES::end(__base_)};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end() const
  {
    if constexpr (sized_range<const _View>)
    {
      if constexpr (random_access_range<const _View>)
      {
        return _CUDA_VRANGES::begin(__base_) + size();
      }
      else
      {
        return default_sentinel;
      }
    }
    else
    {
      return __sentinel<true>{_CUDA_VRANGES::end(__base_)};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(sized_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size()
  {
    const auto __n = _CUDA_VRANGES::size(__base_);
    return (_CUDA_VRANGES::min)(__n, static_cast<decltype(__n)>(__count_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(sized_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size() const
  {
    auto __n = _CUDA_VRANGES::size(__base_);
    return (_CUDA_VRANGES::min)(__n, static_cast<decltype(__n)>(__count_));
  }
};

template <class _Range>
_CCCL_HOST_DEVICE take_view(_Range&&, range_difference_t<_Range>) -> take_view<_CUDA_VIEWS::all_t<_Range>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool enable_borrowed_range<take_view<_Tp>> = enable_borrowed_range<_Tp>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__take)

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_empty_view = false;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_empty_view<empty_view<_Tp>> = true;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_passthrough_specialization = false;

template <class _Tp, size_t _Extent>
_CCCL_INLINE_VAR constexpr bool __is_passthrough_specialization<span<_Tp, _Extent>> = true;

#  if defined(_LIBCUDACXX_HAS_STRING_VIEW)
template <class _CharT, class _Traits>
_CCCL_INLINE_VAR constexpr bool __is_passthrough_specialization<basic_string_view<_CharT, _Traits>> = true;
#  endif // _LIBCUDACXX_HAS_STRING_VIEW

template <class _Iter, class _Sent, _CUDA_VRANGES::subrange_kind _Kind>
_CCCL_INLINE_VAR constexpr bool __is_passthrough_specialization<_CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>> = true;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_iota_specialization = false;

template <class _Np, class _Bound>
_CCCL_INLINE_VAR constexpr bool __is_iota_specialization<iota_view<_Np, _Bound>> = true;

template <class _Tp, class = void>
struct __passthrough_type;

template <class _Tp, size_t _Extent>
struct __passthrough_type<span<_Tp, _Extent>>
{
  using type = span<_Tp>;
};

#  if defined(_LIBCUDACXX_HAS_STRING_VIEW)
template <class _CharT, class _Traits>
struct __passthrough_type<basic_string_view<_CharT, _Traits>>
{
  using type = basic_string_view<_CharT, _Traits>;
};
#  endif //_LIBCUDACXX_HAS_STRING_VIEW

template <class _Iter, class _Sent, _CUDA_VRANGES::subrange_kind _Kind>
struct __passthrough_type<_CUDA_VRANGES::subrange<_Iter, _Sent, _Kind>, void_t<typename _CUDA_VRANGES::subrange<_Iter>>>
{
  using type = _CUDA_VRANGES::subrange<_Iter>;
};

template <class _Tp>
using __passthrough_type_t = typename __passthrough_type<_Tp>::type;

#  if _CCCL_STD_VER >= 2020
template <class _Range, class _Np>
concept __use_empty = convertible_to<_Np, range_difference_t<_Range>> && __is_empty_view<remove_cvref_t<_Range>>;

template <class _Range, class _Np>
concept __use_passthrough =
  convertible_to<_Np, range_difference_t<_Range>> && !__is_empty_view<remove_cvref_t<_Range>>
  && random_access_range<remove_cvref_t<_Range>> && sized_range<remove_cvref_t<_Range>>
  && __is_passthrough_specialization<remove_cvref_t<_Range>>;

template <class _Range, class _Np>
concept __use_iota =
  convertible_to<_Np, range_difference_t<_Range>> && !__is_empty_view<remove_cvref_t<_Range>>
  && random_access_range<remove_cvref_t<_Range>> && sized_range<remove_cvref_t<_Range>>
  && __is_iota_specialization<remove_cvref_t<_Range>>;

template <class _Range, class _Np>
concept __use_generic = !__use_empty<_Range, _Np> && !__is_repeat_specialization<remove_cvref_t<_Range>>
                     && !__use_passthrough<_Range, _Np> && !__use_iota<_Range, _Np>;

#  else // ^^^ C++20 ^^^ / vvv C++17 vvv

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT_FRAGMENT(__use_empty_,
                             requires()(requires(convertible_to<_Np, range_difference_t<_Range>>),
                                        requires(__is_empty_view<remove_cvref_t<_Range>>)));

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT __use_empty = _LIBCUDACXX_FRAGMENT(__use_empty_, _Range, _Np);

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __use_passthrough_,
  requires()(requires(convertible_to<_Np, range_difference_t<_Range>>),
             requires(!__is_empty_view<remove_cvref_t<_Range>>),
             requires(random_access_range<remove_cvref_t<_Range>>),
             requires(sized_range<remove_cvref_t<_Range>>),
             requires(__is_passthrough_specialization<remove_cvref_t<_Range>>)));

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT __use_passthrough = _LIBCUDACXX_FRAGMENT(__use_passthrough_, _Range, _Np);

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __use_iota_,
  requires()(requires(convertible_to<_Np, range_difference_t<_Range>>),
             requires(!__is_empty_view<remove_cvref_t<_Range>>),
             requires(random_access_range<remove_cvref_t<_Range>>),
             requires(sized_range<remove_cvref_t<_Range>>),
             requires(__is_iota_specialization<remove_cvref_t<_Range>>)));

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT __use_iota = _LIBCUDACXX_FRAGMENT(__use_iota_, _Range, _Np);

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __use_generic_,
  requires()(requires(!__use_empty<_Range, _Np>),
             requires(!__is_repeat_specialization<remove_cvref_t<_Range>>),
             requires(!__use_passthrough<_Range, _Np>),
             requires(!__use_iota<_Range, _Np>)));

template <class _Range, class _Np>
_LIBCUDACXX_CONCEPT __use_generic = _LIBCUDACXX_FRAGMENT(__use_generic_, _Range, _Np);
#  endif // _CCCL_STD_VER <= 2017

struct __fn
{
  // [range.take.overview]: the `empty_view` case.
  _LIBCUDACXX_TEMPLATE(class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>)
  _LIBCUDACXX_REQUIRES(__use_empty<_Range, _Np>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, _Np&&) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Range>(__range)))) -> _RawRange
  {
    return _LIBCUDACXX_AUTO_CAST(_CUDA_VSTD::forward<_Range>(__range));
  }

  // [range.take.overview]: the `span | basic_string_view | subrange` case.
  _LIBCUDACXX_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _LIBCUDACXX_REQUIRES(__use_passthrough<_Range, _Np>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __rng, _Np&& __n) const
    noexcept(noexcept(__passthrough_type_t<_RawRange>(
      _CUDA_VRANGES::begin(__rng),
      _CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)))))
      -> __passthrough_type_t<_RawRange>
  {
    return __passthrough_type_t<_RawRange>(
      _CUDA_VRANGES::begin(__rng),
      _CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)));
  }

  // [range.take.overview]: the `repeat_view` "_RawRange models sized_range" case.
  _LIBCUDACXX_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _LIBCUDACXX_REQUIRES(convertible_to<_Np, range_difference_t<_Range>> _LIBCUDACXX_AND
                         __is_repeat_specialization<_RawRange> _LIBCUDACXX_AND sized_range<_RawRange>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator()(_Range&& __range, _Np&& __n) const noexcept(noexcept(views::repeat(
    *__range.__value_, _CUDA_VSTD::min<_Dist>(ranges::distance(__range), _CUDA_VSTD::forward<_Np>(__n))))) -> _RawRange
  {
    return views::repeat(*__range.__value_,
                         _CUDA_VSTD::min<_Dist>(ranges::distance(__range), _CUDA_VSTD::forward<_Np>(__n)));
  }

  // [range.take.overview]: the `repeat_view` "otherwise" case.
  _LIBCUDACXX_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _LIBCUDACXX_REQUIRES(convertible_to<_Np, range_difference_t<_Range>> _LIBCUDACXX_AND
                         __is_repeat_specialization<_RawRange> _LIBCUDACXX_AND(!sized_range<_RawRange>))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, _Np&& __n) const noexcept(
    noexcept(views::repeat(*__range.__value_, static_cast<_Dist>(__n)))) -> repeat_view<range_value_t<_RawRange>, _Dist>
  {
    return views::repeat(*__range.__value_, static_cast<_Dist>(__n));
  }

  // [range.take.overview]: the `iota_view` case.
  _LIBCUDACXX_TEMPLATE(
    class _Range, class _Np, class _RawRange = remove_cvref_t<_Range>, class _Dist = range_difference_t<_Range>)
  _LIBCUDACXX_REQUIRES(__use_iota<_Range, _Np>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __rng, _Np&& __n) const
    noexcept(noexcept(_CUDA_VRANGES::iota_view(
      *_CUDA_VRANGES::begin(__rng),
      *_CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)))))
      -> iota_view<range_value_t<_RawRange>, _Dist>
  {
    return _CUDA_VRANGES::iota_view(
      *_CUDA_VRANGES::begin(__rng),
      *_CUDA_VRANGES::begin(__rng)
        + _CUDA_VSTD::min<_Dist>(_CUDA_VRANGES::distance(__rng), _CUDA_VSTD::forward<_Np>(__n)));
  }

  // [range.take.overview]: the "otherwise" case.
  _LIBCUDACXX_TEMPLATE(class _Range, class _Np)
  _LIBCUDACXX_REQUIRES(__use_generic<_Range, _Np>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range, _Np&& __n) const
    noexcept(noexcept(take_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Np>(__n))))
      -> decltype(take_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Np>(__n)))
  {
    return take_view(_CUDA_VSTD::forward<_Range>(__range), _CUDA_VSTD::forward<_Np>(__n));
  }

  _LIBCUDACXX_TEMPLATE(class _Np)
  _LIBCUDACXX_REQUIRES(constructible_from<decay_t<_Np>, _Np>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Np&& __n) const
    noexcept(is_nothrow_constructible_v<decay_t<_Np>, _Np>)
  {
    return __range_adaptor_closure_t(_CUDA_VSTD::__bind_back(*this, _CUDA_VSTD::forward<_Np>(__n)));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto take = __take::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_CCCL_POP_MACROS

#endif // _LIBCUDACXX___RANGES_TAKE_VIEW_H
