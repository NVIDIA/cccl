// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_SUBRANGE_H
#define _LIBCUDACXX___RANGES_SUBRANGE_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__assert"
#include "../__concepts/constructible.h"
#include "../__concepts/convertible_to.h"
#include "../__concepts/copyable.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/different_from.h"
#include "../__fwd/get.h"
#include "../__fwd/subrange.h"
#include "../__iterator/advance.h"
#include "../__iterator/concepts.h"
#include "../__iterator/incrementable_traits.h"
#include "../__iterator/iterator_traits.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/dangling.h"
#include "../__ranges/enable_borrowed_range.h"
#include "../__ranges/size.h"
#include "../__ranges/view_interface.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__tuple_dir/structured_bindings.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_nothrow_default_constructible.h"
#include "../__type_traits/is_pointer.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/make_unsigned.h"
#include "../__type_traits/remove_const.h"
#include "../__type_traits/remove_pointer.h"
#include "../__utility/move.h"
#include "../cstdlib"

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <class _From, class _To>
concept __uses_nonqualification_pointer_conversion =
  is_pointer_v<_From> && is_pointer_v<_To>
  && !convertible_to<remove_pointer_t<_From> (*)[], remove_pointer_t<_To> (*)[]>;

template <class _From, class _To>
concept __convertible_to_non_slicing =
  convertible_to<_From, _To> && !__uses_nonqualification_pointer_conversion<decay_t<_From>, decay_t<_To>>;

template <class _Tp>
concept __pair_like = !is_reference_v<_Tp> && requires(_Tp __t) {
  typename tuple_size<_Tp>::type; // Ensures `tuple_size<T>` is complete.
  requires derived_from<tuple_size<_Tp>, integral_constant<size_t, 2>>;
  typename tuple_element_t<0, remove_const_t<_Tp>>;
  typename tuple_element_t<1, remove_const_t<_Tp>>;
  {
    _CUDA_VSTD::get<0>(__t)
  } -> convertible_to<const tuple_element_t<0, _Tp>&>;
  {
    _CUDA_VSTD::get<1>(__t)
  } -> convertible_to<const tuple_element_t<1, _Tp>&>;
};

template <class _Pair, class _Iter, class _Sent>
concept __pair_like_convertible_from =
  !range<_Pair> && __pair_like<_Pair> && constructible_from<_Pair, _Iter, _Sent>
  && __convertible_to_non_slicing<_Iter, tuple_element_t<0, _Pair>> && convertible_to<_Sent, tuple_element_t<1, _Pair>>;

// We have issues with MSVC and _StoreSize being unable to be properly determined in SFINAE, so we need to pull that out
template <class _Iter, class _It, bool _StoreSize>
concept __subrange_from_iter_sent = !_StoreSize && __convertible_to_non_slicing<_It, _Iter>;

template <class _Iter, subrange_kind _Kind, class _It>
concept __subrange_from_iter_sent_size = _Kind == subrange_kind::sized && __convertible_to_non_slicing<_It, _Iter>;

template <class _Iter, class _Sent, subrange_kind _Kind, class _Range, bool _StoreSize>
concept __subrange_from_range =
  _StoreSize && __different_from<_Range, subrange<_Iter, _Sent, _Kind>> && borrowed_range<_Range>
  && __convertible_to_non_slicing<iterator_t<_Range>, _Iter> && convertible_to<sentinel_t<_Range>, _Sent>;

template <class _Iter, class _Sent, subrange_kind _Kind, class _Range>
concept __subrange_from_range_size =
  _Kind == subrange_kind::sized && borrowed_range<_Range> && __convertible_to_non_slicing<iterator_t<_Range>, _Iter>
  && convertible_to<sentinel_t<_Range>, _Sent>;

template <class _Iter, class _Sent, subrange_kind _Kind, class _Pair>
concept __subrange_to_pair = __different_from<_Pair, subrange<_Iter, _Sent, _Kind>>
                          && __pair_like_convertible_from<_Pair, const _Iter&, const _Sent&>;

#  else // ^^^ C++20 ^^^ / vvv C++17 vvv

template <class _From, class _To>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __uses_nonqualification_pointer_conversion_,
  requires()(requires(is_pointer_v<_From>),
             requires(is_pointer_v<_To>),
             requires(!convertible_to<remove_pointer_t<_From> (*)[], remove_pointer_t<_To> (*)[]>)));

template <class _From, class _To>
_LIBCUDACXX_CONCEPT __uses_nonqualification_pointer_conversion =
  _LIBCUDACXX_FRAGMENT(__uses_nonqualification_pointer_conversion_, _From, _To);

template <class _From, class _To>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __convertible_to_non_slicing_,
  requires()(requires(convertible_to<_From, _To>),
             requires(!__uses_nonqualification_pointer_conversion<decay_t<_From>, decay_t<_To>>)));

template <class _From, class _To>
_LIBCUDACXX_CONCEPT __convertible_to_non_slicing = _LIBCUDACXX_FRAGMENT(__convertible_to_non_slicing_, _From, _To);

template <class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __pair_like_,
  requires(_Tp __t)(
    requires(!is_reference_v<_Tp>),
    typename(typename tuple_size<_Tp>::type),
    requires(tuple_size<_Tp>::value == 2), // relaxed because of gcc issues
    typename(tuple_element_t<0, remove_const_t<_Tp>>),
    typename(tuple_element_t<1, remove_const_t<_Tp>>),
    requires(convertible_to<decltype(_CUDA_VSTD::get<0>(__t)), const tuple_element_t<0, _Tp>&>),
    requires(convertible_to<decltype(_CUDA_VSTD::get<1>(__t)), const tuple_element_t<1, _Tp>&>)));

template <class _Tp>
_LIBCUDACXX_CONCEPT __pair_like = _LIBCUDACXX_FRAGMENT(__pair_like_, _Tp);

template <class _Pair, class _Iter, class _Sent>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __pair_like_convertible_from_,
  requires()(requires(!range<_Pair>),
             requires(__pair_like<_Pair>),
             requires(constructible_from<_Pair, _Iter, _Sent>),
             requires(__convertible_to_non_slicing<_Iter, tuple_element_t<0, _Pair>>),
             requires(convertible_to<_Sent, tuple_element_t<1, _Pair>>)));

template <class _Pair, class _Iter, class _Sent>
_LIBCUDACXX_CONCEPT __pair_like_convertible_from =
  _LIBCUDACXX_FRAGMENT(__pair_like_convertible_from_, _Pair, _Iter, _Sent);

// We have issues with MSVC and _StoreSize being unable to be properly determined in SFINAE, so we need to pull that out
template <class _Iter, class _It, class _StoreSize>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __subrange_from_iter_sent_,
  requires()(requires(!_StoreSize::value), requires(__convertible_to_non_slicing<_It, _Iter>)));

template <class _Iter, class _It, bool _StoreSize>
_LIBCUDACXX_CONCEPT __subrange_from_iter_sent =
  _LIBCUDACXX_FRAGMENT(__subrange_from_iter_sent_, _Iter, _It, integral_constant<bool, _StoreSize>);

template <class _Iter, class _Kind, class _It>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __subrange_from_iter_sent_size_,
  requires()(requires(_Kind::value == subrange_kind::sized), requires(__convertible_to_non_slicing<_It, _Iter>)));

template <class _Iter, subrange_kind _Kind, class _It>
_LIBCUDACXX_CONCEPT __subrange_from_iter_sent_size =
  _LIBCUDACXX_FRAGMENT(__subrange_from_iter_sent_size_, _Iter, integral_constant<subrange_kind, _Kind>, _It);

template <class _Iter, class _Sent, class _Kind, class _Range, class _StoreSize>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __subrange_from_range_,
  requires()(requires(_StoreSize::value),
             requires(__different_from<_Range, subrange<_Iter, _Sent, _Kind::value>>),
             requires(borrowed_range<_Range>),
             requires(__convertible_to_non_slicing<iterator_t<_Range>, _Iter>),
             requires(convertible_to<sentinel_t<_Range>, _Sent>)));

template <class _Iter, class _Sent, subrange_kind _Kind, class _Range, bool _StoreSize>
_LIBCUDACXX_CONCEPT __subrange_from_range = _LIBCUDACXX_FRAGMENT(
  __subrange_from_range_,
  _Iter,
  _Sent,
  integral_constant<subrange_kind, _Kind>,
  _Range,
  integral_constant<bool, _StoreSize>);

template <class _Iter, class _Sent, class _Kind, class _Range>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __subrange_from_range_size_,
  requires()(requires((_Kind::value == subrange_kind::sized)),
             requires(borrowed_range<_Range>),
             requires(__convertible_to_non_slicing<iterator_t<_Range>, _Iter>),
             requires(convertible_to<sentinel_t<_Range>, _Sent>)));

template <class _Iter, class _Sent, subrange_kind _Kind, class _Range>
_LIBCUDACXX_CONCEPT __subrange_from_range_size =
  _LIBCUDACXX_FRAGMENT(__subrange_from_range_size_, _Iter, _Sent, integral_constant<subrange_kind, _Kind>, _Range);

template <class _Iter, class _Sent, class _Kind, class _Pair>
_LIBCUDACXX_CONCEPT_FRAGMENT(__subrange_to_pair_,
                             requires()(requires(__different_from<_Pair, subrange<_Iter, _Sent, _Kind::value>>),
                                        requires(__pair_like_convertible_from<_Pair, const _Iter&, const _Sent&>)));

template <class _Iter, class _Sent, subrange_kind _Kind, class _Pair>
_LIBCUDACXX_CONCEPT __subrange_to_pair =
  _LIBCUDACXX_FRAGMENT(__subrange_to_pair_, _Iter, _Sent, integral_constant<subrange_kind, _Kind>, _Pair);
#  endif // _CCCL_STD_VER <= 2017

#  if _CCCL_STD_VER >= 2020
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent, subrange_kind _Kind>
  requires(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>)
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter,
          class _Sent,
          subrange_kind _Kind,
          enable_if_t<input_or_output_iterator<_Iter>, int>,
          enable_if_t<sentinel_for<_Sent, _Iter>, int>,
          enable_if_t<(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>), int>>
#  endif // _CCCL_STD_VER <= 2017
class _LIBCUDACXX_TEMPLATE_VIS subrange : public view_interface<subrange<_Iter, _Sent, _Kind>>
{
public:
  // Note: this is an internal implementation detail that is public only for internal usage.
  static constexpr bool _StoreSize = (_Kind == subrange_kind::sized && !sized_sentinel_for<_Sent, _Iter>);

private:
  struct _Empty
  {
    template <class _Tp>
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Empty(_Tp) noexcept
    {}
  };
  using _Size = conditional_t<_StoreSize, make_unsigned_t<iter_difference_t<_Iter>>, _Empty>;
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Iter __begin_ = _Iter();
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Sent __end_   = _Sent();
  _LIBCUDACXX_NO_UNIQUE_ADDRESS _Size __size_  = 0;

public:
#  if _CCCL_STD_VER >= 2020
  subrange()
    requires default_initializable<_Iter>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  template <class _It = _Iter, enable_if_t<default_initializable<_It>, int> = 0>
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange() noexcept(
    is_nothrow_default_constructible_v<_It>)
      : view_interface<subrange<_Iter, _Sent, _Kind>>(){};
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_TEMPLATE(class _It)
  _LIBCUDACXX_REQUIRES(__subrange_from_iter_sent<_Iter, _It, _StoreSize>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange(_It __iter, _Sent __sent)
      : view_interface<subrange<_Iter, _Sent, _Kind>>()
      , __begin_(_CUDA_VSTD::move(__iter))
      , __end_(_CUDA_VSTD::move(__sent))
  {}

  _LIBCUDACXX_TEMPLATE(class _It)
  _LIBCUDACXX_REQUIRES(__subrange_from_iter_sent_size<_Iter, _Kind, _It>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange(
    _It __iter, _Sent __sent, make_unsigned_t<iter_difference_t<_Iter>> __n)
      : view_interface<subrange<_Iter, _Sent, _Kind>>()
      , __begin_(_CUDA_VSTD::move(__iter))
      , __end_(_CUDA_VSTD::move(__sent))
      , __size_(__n)
  {
    if constexpr (sized_sentinel_for<_Sent, _Iter>)
    {
      _LIBCUDACXX_ASSERT((__end_ - __begin_) == static_cast<iter_difference_t<_Iter>>(__n),
                         "_CUDA_VSTD::_CUDA_VRANGES::subrange was passed an invalid size hint");
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__subrange_from_range<_Iter, _Sent, _Kind, _Range, !_StoreSize>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange(_Range&& __range)
      : subrange(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::end(__range))
  {}

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__subrange_from_range<_Iter, _Sent, _Kind, _Range, _StoreSize>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange(_Range&& __range)
      : subrange(__range, _CUDA_VRANGES::size(__range))
  {}

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__subrange_from_range_size<_Iter, _Sent, _Kind, _Range>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange(
    _Range&& __range, make_unsigned_t<iter_difference_t<_Iter>> __n)
      : subrange(_CUDA_VRANGES::begin(__range), _CUDA_VRANGES::end(__range), __n)
  {}

#  if (!defined(_CCCL_COMPILER_GCC) || _GNUC_VER >= 900)
  _LIBCUDACXX_TEMPLATE(class _Pair)
  _LIBCUDACXX_REQUIRES(__pair_like<_Pair> _LIBCUDACXX_AND __subrange_to_pair<_Iter, _Sent, _Kind, _Pair>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr operator _Pair() const
  {
    return _Pair(__begin_, __end_);
  }
#  endif // !(_CCCL_COMPILER_GCC < 9)

  _LIBCUDACXX_TEMPLATE(class _It = _Iter)
  _LIBCUDACXX_REQUIRES(copyable<_It>)
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _It begin() const
  {
    return __begin_;
  }

  _LIBCUDACXX_TEMPLATE(class _It = _Iter)
  _LIBCUDACXX_REQUIRES((!copyable<_It>) )
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _It begin()
  {
    return _CUDA_VSTD::move(__begin_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr _Sent end() const
  {
    return __end_;
  }

  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr bool empty() const
  {
    return __begin_ == __end_;
  }

  _LIBCUDACXX_TEMPLATE(subrange_kind _Kind_ = _Kind)
  _LIBCUDACXX_REQUIRES((_Kind_ == subrange_kind::sized))
  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr make_unsigned_t<iter_difference_t<_Iter>>
  size() const
  {
    if constexpr (_StoreSize)
    {
      return __size_;
    }
    else
    {
      return _CUDA_VSTD::__to_unsigned_like(__end_ - __begin_);
    }
#  if defined(_LIBCUDACXX_CUDACC_BELOW_11_3)
    _LIBCUDACXX_UNREACHABLE();
#  endif // _LIBCUDACXX_CUDACC_BELOW_11_3
  }

  _LIBCUDACXX_TEMPLATE(class _It = _Iter)
  _LIBCUDACXX_REQUIRES(forward_iterator<_It>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange
  next(iter_difference_t<_Iter> __n = 1) const&
  {
    auto __tmp = *this;
    __tmp.advance(__n);
    return __tmp;
  }

  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange
  next(iter_difference_t<_Iter> __n = 1) &&
  {
    advance(__n);
    return _CUDA_VSTD::move(*this);
  }

  _LIBCUDACXX_TEMPLATE(class _It = _Iter)
  _LIBCUDACXX_REQUIRES(bidirectional_iterator<_It>)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange
  prev(iter_difference_t<_Iter> __n = 1) const
  {
    auto __tmp = *this;
    __tmp.advance(-__n);
    return __tmp;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr subrange& advance(iter_difference_t<_Iter> __n)
  {
    if constexpr (bidirectional_iterator<_Iter>)
    {
      if (__n < 0)
      {
        _CUDA_VRANGES::advance(__begin_, __n);
        if constexpr (_StoreSize)
        {
          __size_ += _CUDA_VSTD::__to_unsigned_like(-__n);
        }
        return *this;
      }
    }

    const auto __d = __n - _CUDA_VRANGES::advance(__begin_, __n, __end_);
    if constexpr (_StoreSize)
    {
      __size_ -= _CUDA_VSTD::__to_unsigned_like(__d);
    }
    (void)__d;
    return *this;
  }
};

_LIBCUDACXX_TEMPLATE(class _Iter, class _Sent)
_LIBCUDACXX_REQUIRES(input_or_output_iterator<_Iter> _LIBCUDACXX_AND sentinel_for<_Sent, _Iter>)
_CCCL_HOST_DEVICE subrange(_Iter, _Sent) -> subrange<_Iter, _Sent>;

_LIBCUDACXX_TEMPLATE(class _Iter, class _Sent)
_LIBCUDACXX_REQUIRES(input_or_output_iterator<_Iter> _LIBCUDACXX_AND sentinel_for<_Sent, _Iter>)
_CCCL_HOST_DEVICE subrange(_Iter, _Sent, make_unsigned_t<iter_difference_t<_Iter>>)
  -> subrange<_Iter, _Sent, subrange_kind::sized>;

_LIBCUDACXX_TEMPLATE(class _Range)
_LIBCUDACXX_REQUIRES(borrowed_range<_Range>)
_CCCL_HOST_DEVICE subrange(_Range&&)
  -> subrange<iterator_t<_Range>,
              sentinel_t<_Range>,
              (sized_range<_Range> || sized_sentinel_for<sentinel_t<_Range>, iterator_t<_Range>>)
                ? subrange_kind::sized
                : subrange_kind::unsized>;

_LIBCUDACXX_TEMPLATE(class _Range)
_LIBCUDACXX_REQUIRES(borrowed_range<_Range>)
_CCCL_HOST_DEVICE subrange(_Range&&, make_unsigned_t<range_difference_t<_Range>>)
  -> subrange<iterator_t<_Range>, sentinel_t<_Range>, subrange_kind::sized>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

// Not _LIBCUDACXX_TEMPLATE because we need to forward declare them
#  if _CCCL_STD_VER >= 2020
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires((_Index == 0) && copyable<_Iter>) || (_Index == 1)
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <size_t _Index,
          class _Iter,
          class _Sent,
          subrange_kind _Kind,
          enable_if_t<((_Index == 0) && copyable<_Iter>) || (_Index == 1), int>>
#  endif // _CCCL_STD_VER <= 2017
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto get(
  const subrange<_Iter, _Sent, _Kind>& __subrange)
{
  if constexpr (_Index == 0)
  {
    return __subrange.begin();
  }
  else
  {
    return __subrange.end();
  }
  _LIBCUDACXX_UNREACHABLE();
}

#  if _CCCL_STD_VER >= 2020
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires(_Index < 2)
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <
  size_t _Index,
  class _Iter,
  class _Sent,
  subrange_kind _Kind,
  enable_if_t<_Index< 2, int>>
#  endif // _CCCL_STD_VER <= 2017
_LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY constexpr auto get(subrange<_Iter, _Sent, _Kind>&& __subrange)
{
  if constexpr (_Index == 0)
  {
    return __subrange.begin();
  }
  else
  {
    return __subrange.end();
  }
  _LIBCUDACXX_UNREACHABLE();
}

template <class _Ip, class _Sp, subrange_kind _Kp>
_LIBCUDACXX_INLINE_VAR constexpr bool enable_borrowed_range<subrange<_Ip, _Sp, _Kp>> = true;

template <class _Rp>
using borrowed_subrange_t = enable_if_t<range<_Rp>, _If<borrowed_range<_Rp>, subrange<iterator_t<_Rp>>, dangling>>;

_LIBCUDACXX_END_NAMESPACE_RANGES

// [range.subrange.general]

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using _CUDA_VRANGES::get;

// [ranges.syn]

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_size<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>> : integral_constant<size_t, 2>
{};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<0, _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Ip;
};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<1, _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Sp;
};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<0, const _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Ip;
};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<1, const _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{
  using type = _Sp;
};

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___RANGES_SUBRANGE_H
