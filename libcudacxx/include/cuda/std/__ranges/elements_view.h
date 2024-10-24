//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_ELEMENTS_VIEW_H
#define _LIBCUDACXX___RANGES_ELEMENTS_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

template <class _View, size_t _Np, bool _Const>
class __elements_view_iterator;

template <class _View, size_t _Np, bool _Const>
class __elements_view_sentinel;

#  if _CCCL_STD_VER >= 2020
template <class _Tp, size_t _Np>
concept __has_tuple_element = __tuple_like<_Tp>::value && (_Np < tuple_size_v<_Tp>);
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp, class _Np>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __has_tuple_element_, requires()(requires(__tuple_like<_Tp>::value), requires(_Np::value < tuple_size<_Tp>::value)));

template <class _Tp, size_t _Np>
_LIBCUDACXX_CONCEPT __has_tuple_element =
  _LIBCUDACXX_FRAGMENT(__has_tuple_element_, _Tp, integral_constant<size_t, _Np>);
#  endif // _CCCL_STD_VER <= 2017

template <class _Tp, size_t _Np, class = void>
_CCCL_INLINE_VAR constexpr bool __returnable_element = is_reference_v<_Tp>;

template <class _Tp, size_t _Np>
_CCCL_INLINE_VAR constexpr bool
  __returnable_element<_Tp, _Np, enable_if_t<move_constructible<tuple_element_t<_Np, _Tp>>>> = true;

#  if _CCCL_STD_VER >= 2020
template <input_range _View, size_t _Np>
  requires view<_View> && __has_tuple_element<range_value_t<_View>, _Np>
        && __has_tuple_element<remove_reference_t<range_reference_t<_View>>, _Np>
        && __returnable_element<range_reference_t<_View>, _Np>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _View,
          size_t _Np,
          enable_if_t<input_range<_View>, int>                                                     = 0,
          enable_if_t<view<_View>, int>                                                            = 0,
          enable_if_t<__has_tuple_element<range_value_t<_View>, _Np>, int>                         = 0,
          enable_if_t<__has_tuple_element<remove_reference_t<range_reference_t<_View>>, _Np>, int> = 0,
          enable_if_t<__returnable_element<range_reference_t<_View>, _Np>, int>                    = 0>
#  endif // _CCCL_STD_VER <= 2017
class elements_view : public view_interface<elements_view<_View, _Np>>
{
private:
  template <bool _Const>
  using __iterator = __elements_view_iterator<_View, _Np, _Const>;

  template <bool _Const>
  using __sentinel = __elements_view_sentinel<_View, _Np, _Const>;

  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();

public:
#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI elements_view()
    requires default_initializable<_View>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(default_initializable<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr elements_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<elements_view<_View, _Np>>()
  {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit elements_view(_View __base) noexcept(
    is_nothrow_move_constructible_v<_View>)
      : view_interface<elements_view<_View, _Np>>()
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

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES((!__simple_view<_View2>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin()
  {
    return __iterator</*_Const=*/false>(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin() const
  {
    return __iterator</*_Const=*/true>(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES((!__simple_view<_View2>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end()
  {
    if constexpr (common_range<_View>)
    {
      return __iterator</*_Const=*/false>{_CUDA_VRANGES::end(__base_)};
    }
    else
    {
      return __sentinel</*_Const=*/false>{_CUDA_VRANGES::end(__base_)};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end() const
  {
    if constexpr (common_range<_View>)
    {
      return __iterator</*_Const=*/true>{_CUDA_VRANGES::end(__base_)};
    }
    else
    {
      return __sentinel</*_Const=*/true>{_CUDA_VRANGES::end(__base_)};
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(sized_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size()
  {
    return _CUDA_VRANGES::size(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(sized_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto size() const
  {
    return _CUDA_VRANGES::size(__base_);
  }
};

template <class, size_t, class = void>
struct __elements_view_iterator_category_base
{};

template <class _Base, size_t _Np>
struct __elements_view_iterator_category_base<_Base, _Np, enable_if_t<forward_range<_Base>>>
{
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __get_iterator_category() noexcept
  {
    using _Result = decltype(_CUDA_VSTD::get<_Np>(*_CUDA_VSTD::declval<iterator_t<_Base>>()));
    using _Cat    = typename iterator_traits<iterator_t<_Base>>::iterator_category;

    if constexpr (!is_lvalue_reference_v<_Result>)
    {
      return input_iterator_tag{};
    }
    else if constexpr (derived_from<_Cat, random_access_iterator_tag>)
    {
      return random_access_iterator_tag{};
    }
    else
    {
      return _Cat{};
    }
    _CCCL_UNREACHABLE();
  }

  using iterator_category = decltype(__get_iterator_category());
};

template <class _View, size_t _Np, bool _Const>
class __elements_view_iterator : public __elements_view_iterator_category_base<__maybe_const<_Const, _View>, _Np>
{
  template <class, size_t, bool>
  friend class __elements_view_iterator;

  template <class, size_t, bool>
  friend class __elements_view_sentinel;

  using _Base = __maybe_const<_Const, _View>;
  template <bool _OtherConst>
  using _Base2 = __maybe_const<_OtherConst, _View>;

  iterator_t<_Base> __current_ = iterator_t<_Base>();

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr decltype(auto) __get_element(const iterator_t<_Base>& __i)
  {
    if constexpr (is_reference_v<range_reference_t<_Base>>)
    {
      return _CUDA_VSTD::get<_Np>(*__i);
    }
    else
    {
      using _Element = remove_cv_t<tuple_element_t<_Np, range_reference_t<_Base>>>;
#  if defined(_CCCL_COMPILER_MSVC) // MSVC does not copy with the static_cast
      return _Element(_CUDA_VSTD::get<_Np>(*__i));
#  else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
      return static_cast<_Element>(_CUDA_VSTD::get<_Np>(*__i));
#  endif // !_CCCL_COMPILER_MSVC
    }
    _CCCL_UNREACHABLE();
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __get_iterator_concept() noexcept
  {
    if constexpr (random_access_range<_Base>)
    {
      return random_access_iterator_tag{};
    }
    else if constexpr (bidirectional_range<_Base>)
    {
      return bidirectional_iterator_tag{};
    }
    else if constexpr (forward_range<_Base>)
    {
      return forward_iterator_tag{};
    }
    else
    {
      return input_iterator_tag{};
    }
    _CCCL_UNREACHABLE();
  }

public:
  using iterator_concept = decltype(__get_iterator_concept());
  using value_type       = remove_cvref_t<tuple_element_t<_Np, range_value_t<_Base>>>;
  using difference_type  = range_difference_t<_Base>;

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI __elements_view_iterator()
    requires default_initializable<iterator_t<_Base>>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(default_initializable<iterator_t<_Base2<_Const2>>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator() noexcept(
    is_nothrow_default_constructible_v<iterator_t<_Base2<_Const2>>>)
  {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __elements_view_iterator(iterator_t<_Base> __current) noexcept(
    is_nothrow_move_constructible_v<iterator_t<_Base>>)
      : __current_(_CUDA_VSTD::move(__current))
  {}

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(_Const2 _LIBCUDACXX_AND convertible_to<iterator_t<_View>, iterator_t<_Base>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator(__elements_view_iterator<_View, _Np, !_Const2> __i)
      : __current_(_CUDA_VSTD::move(__i.__current_))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const iterator_t<_Base>& base() const& noexcept
  {
    return __current_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr iterator_t<_Base> base() &&
  {
    return _CUDA_VSTD::move(__current_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator*() const
  {
    return __get_element(__current_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator& operator++()
  {
    ++__current_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES((!forward_range<_Base2<_Const2>>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr void operator++(int)
  {
    ++__current_;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(forward_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator operator++(int)
  {
    auto temp = *this;
    ++__current_;
    return temp;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(bidirectional_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator& operator--()
  {
    --__current_;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(bidirectional_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator operator--(int)
  {
    auto temp = *this;
    --__current_;
    return temp;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(random_access_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator& operator+=(difference_type __n)
  {
    __current_ += __n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(random_access_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_iterator& operator-=(difference_type __n)
  {
    __current_ -= __n;
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(random_access_range<_Base2<_Const2>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
  {
    return __get_element(__current_ + __n);
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator==(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(equality_comparable<iterator_t<_Base2<_Const2>>>)
  {
    return __x.__current_ == __y.__current_;
  }
#  if _CCCL_STD_VER <= 2017
  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator!=(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(equality_comparable<iterator_t<_Base2<_Const2>>>)
  {
    return __x.__current_ != __y.__current_;
  }
#  endif // _CCCL_STD_VER <= 2017

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator<(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(random_access_range<_Base2<_Const2>>)
  {
    return __x.__current_ < __y.__current_;
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator>(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(random_access_range<_Base2<_Const2>>)
  {
    return __y < __x;
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator<=(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(random_access_range<_Base2<_Const2>>)
  {
    return !(__y < __x);
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator>=(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(random_access_range<_Base2<_Const2>>)
  {
    return !(__x < __y);
  }

#  ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr auto
  operator<=>(const __elements_view_iterator& __x, const __elements_view_iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_ <=> __y.__current_;
  }
#  endif // !_LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator+(const __elements_view_iterator& __x, difference_type __y)
    _LIBCUDACXX_TRAILING_REQUIRES(__elements_view_iterator)(random_access_range<_Base2<_Const2>>)
  {
    return __elements_view_iterator{__x} += __y;
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator+(difference_type __x, const __elements_view_iterator& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(__elements_view_iterator)(random_access_range<_Base2<_Const2>>)
  {
    return __y + __x;
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator-(const __elements_view_iterator& __x, difference_type __y)
    _LIBCUDACXX_TRAILING_REQUIRES(__elements_view_iterator)(random_access_range<_Base2<_Const2>>)
  {
    return __elements_view_iterator{__x} -= __y;
  }

  template <bool _Const2 = _Const>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator-(const __elements_view_iterator& __x, const __elements_view_iterator& __y) _LIBCUDACXX_TRAILING_REQUIRES(
    difference_type)(sized_sentinel_for<iterator_t<_Base2<_Const2>>, iterator_t<_Base2<_Const2>>>)
  {
    return __x.__current_ - __y.__current_;
  }
};

template <class _View, size_t _Np, bool _Const>
class __elements_view_sentinel
{
private:
  using _Base                                      = __maybe_const<_Const, _View>;
  _CCCL_NO_UNIQUE_ADDRESS sentinel_t<_Base> __end_ = sentinel_t<_Base>();

  template <class, size_t, bool>
  friend class __elements_view_sentinel;

  template <bool _AnyConst>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr decltype(auto)
  __get_current(const __elements_view_iterator<_View, _Np, _AnyConst>& __iter)
  {
    return (__iter.__current_);
  }

public:
  _CCCL_HIDE_FROM_ABI __elements_view_sentinel() = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __elements_view_sentinel(sentinel_t<_Base> __end)
      : __end_(_CUDA_VSTD::move(__end))
  {}

  _LIBCUDACXX_TEMPLATE(bool _Const2 = _Const)
  _LIBCUDACXX_REQUIRES(_Const2 _LIBCUDACXX_AND convertible_to<sentinel_t<_View>, sentinel_t<_Base>>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __elements_view_sentinel(__elements_view_sentinel<_View, _Np, !_Const2> __other)
      : __end_(_CUDA_VSTD::move(__other.__end_))
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr sentinel_t<_Base> base() const
  {
    return __end_;
  }

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator==(const __elements_view_iterator<_View, _Np, _OtherConst>& __x, const __elements_view_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
  {
    return __get_current(__x) == __y.__end_;
  }
#  if _CCCL_STD_VER <= 2017
  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator==(const __elements_view_sentinel& __y, const __elements_view_iterator<_View, _Np, _OtherConst>& __x)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
  {
    return __get_current(__x) == __y.__end_;
  }
  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator!=(const __elements_view_iterator<_View, _Np, _OtherConst>& __x, const __elements_view_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
  {
    return __get_current(__x) != __y.__end_;
  }
  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator!=(const __elements_view_sentinel& __y, const __elements_view_iterator<_View, _Np, _OtherConst>& __x)
    _LIBCUDACXX_TRAILING_REQUIRES(bool)(sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>)
  {
    return __get_current(__x) != __y.__end_;
  }
#  endif // _CCCL_STD_VER <= 2017

  template <bool _OtherConst>
  static constexpr bool __sized_sentinel =
    sized_sentinel_for<sentinel_t<_Base>, iterator_t<__maybe_const<_OtherConst, _View>>>;

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator-(const __elements_view_iterator<_View, _Np, _OtherConst>& __x, const __elements_view_sentinel& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(range_difference_t<__maybe_const<_OtherConst, _View>>)(__sized_sentinel<_OtherConst>)
  {
    return __get_current(__x) - __y.__end_;
  }

  template <bool _OtherConst>
  friend _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator-(const __elements_view_sentinel& __x, const __elements_view_iterator<_View, _Np, _OtherConst>& __y)
    _LIBCUDACXX_TRAILING_REQUIRES(range_difference_t<__maybe_const<_OtherConst, _View>>)(__sized_sentinel<_OtherConst>)
  {
    return __x.__end_ - __get_current(__y);
  }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _Tp, size_t _Np>
_CCCL_INLINE_VAR constexpr bool enable_borrowed_range<elements_view<_Tp, _Np>> = enable_borrowed_range<_Tp>;

template <class _Tp>
using keys_view = elements_view<_Tp, 0>;
template <class _Tp>
using values_view = elements_view<_Tp, 1>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__elements)

template <size_t _Np>
struct __fn : __range_adaptor_closure<__fn<_Np>>
{
  template <class _Range>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(/**/ elements_view<all_t<_Range&&>, _Np>(_CUDA_VSTD::forward<_Range>(__range))))
      -> elements_view<all_t<_Range>, _Np>
  {
    return /*-----------*/ elements_view<all_t<_Range&&>, _Np>(_CUDA_VSTD::forward<_Range>(__range));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
#  if defined(_CCCL_COMPILER_MSVC)
template <size_t _Np>
_CCCL_INLINE_VAR constexpr auto elements = __elements::__fn<_Np>{};
#  else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
template <size_t _Np>
_CCCL_GLOBAL_CONSTANT auto elements = __elements::__fn<_Np>{};
#  endif // !_CCCL_COMPILER_MSVC
_CCCL_GLOBAL_CONSTANT auto keys   = elements<0>;
_CCCL_GLOBAL_CONSTANT auto values = elements<1>;
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___RANGES_ELEMENTS_VIEW_H
