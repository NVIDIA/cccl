// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_REVERSE_VIEW_H
#define _LIBCUDACXX___RANGES_REVERSE_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/non_propagating_cache.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/subrange.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <view _View>
  requires bidirectional_range<_View>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _View, class = enable_if_t<view<_View>>, class = enable_if_t<bidirectional_range<_View>>>
#  endif // _CCCL_STD_VER <= 2017
class reverse_view : public view_interface<reverse_view<_View>>
{
  // We cache begin() whenever ranges::next is not guaranteed O(1) to provide an
  // amortized O(1) begin() method.
  static constexpr bool _UseCache = !random_access_range<_View> && !common_range<_View>;
  using _Cache = _If<_UseCache, __non_propagating_cache<reverse_iterator<iterator_t<_View>>>, __empty_cache>;
  _CCCL_NO_UNIQUE_ADDRESS _Cache __cached_begin_ = _Cache();
  _CCCL_NO_UNIQUE_ADDRESS _View __base_          = _View();

public:
#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI reverse_view()
    requires default_initializable<_View>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(default_initializable<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reverse_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<reverse_view<_View>>()
  {}
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit reverse_view(_View __view)
      : __base_(_CUDA_VSTD::move(__view))
  {}

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(copy_constructible<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View>)
  {
    return __base_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>)
  {
    return _CUDA_VSTD::move(__base_);
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES((!common_range<_View2>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reverse_iterator<iterator_t<_View2>> begin()
  {
    if constexpr (_UseCache)
    {
      if (__cached_begin_.__has_value())
      {
        return *__cached_begin_;
      }
    }

    auto __tmp = _CUDA_VSTD::make_reverse_iterator(
      _CUDA_VRANGES::next(_CUDA_VRANGES::begin(__base_), _CUDA_VRANGES::end(__base_)));
    if constexpr (_UseCache)
    {
      __cached_begin_.__emplace(__tmp);
    }
    return __tmp;
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(common_range<_View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr reverse_iterator<iterator_t<_View2>> begin()
  {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::end(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(common_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto begin() const
  {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::end(__base_));
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reverse_iterator<iterator_t<_View>> end()
  {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::begin(__base_));
  }

  _LIBCUDACXX_TEMPLATE(class _View2 = _View)
  _LIBCUDACXX_REQUIRES(common_range<const _View2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto end() const
  {
    return _CUDA_VSTD::make_reverse_iterator(_CUDA_VRANGES::begin(__base_));
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

template <class _Range>
_CCCL_HOST_DEVICE reverse_view(_Range&&) -> reverse_view<_CUDA_VIEWS::all_t<_Range>>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool enable_borrowed_range<reverse_view<_Tp>> = enable_borrowed_range<_Tp>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__reverse)

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_reverse_view = false;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_reverse_view<reverse_view<_Tp>> = true;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_sized_reverse_subrange = false;

template <class _Iter>
_CCCL_INLINE_VAR constexpr bool __is_sized_reverse_subrange<
  _CUDA_VRANGES::subrange<reverse_iterator<_Iter>, reverse_iterator<_Iter>, _CUDA_VRANGES::subrange_kind::sized>> =
  true;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_unsized_reverse_subrange = false;

template <class _Iter, subrange_kind _Kind>
_CCCL_INLINE_VAR constexpr bool
  __is_unsized_reverse_subrange<_CUDA_VRANGES::subrange<reverse_iterator<_Iter>, reverse_iterator<_Iter>, _Kind>> =
    _Kind == _CUDA_VRANGES::subrange_kind::unsized;

template <class _Tp>
struct __unwrapped_reverse_subrange
{
  using type = void; // avoid SFINAE-ing out the overload below -- let the concept requirements do it for better
                     // diagnostics
};

template <class _Iter, _CUDA_VRANGES::subrange_kind _Kind>
struct __unwrapped_reverse_subrange<_CUDA_VRANGES::subrange<reverse_iterator<_Iter>, reverse_iterator<_Iter>, _Kind>>
{
  using type = _CUDA_VRANGES::subrange<_Iter, _Iter, _Kind>;
};

template <class _Tp>
using __unwrapped_reverse_subrange_t = typename __unwrapped_reverse_subrange<_Tp>::type;

struct __fn : __range_adaptor_closure<__fn>
{
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__is_reverse_view<remove_cvref_t<_Range>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const noexcept(
    noexcept(_CUDA_VSTD::forward<_Range>(__range).base())) -> decltype(_CUDA_VSTD::forward<_Range>(__range).base())
  {
    return _CUDA_VSTD::forward<_Range>(__range).base();
  }

  _LIBCUDACXX_TEMPLATE(class _Range, class _UnwrappedSubrange = __unwrapped_reverse_subrange_t<remove_cvref_t<_Range>>)
  _LIBCUDACXX_REQUIRES(__is_sized_reverse_subrange<remove_cvref_t<_Range>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const noexcept(
    noexcept(_UnwrappedSubrange(__range.end().base(), __range.begin().base(), __range.size()))) -> _UnwrappedSubrange
  {
    return _UnwrappedSubrange(__range.end().base(), __range.begin().base(), __range.size());
  }

  _LIBCUDACXX_TEMPLATE(class _Range, class _UnwrappedSubrange = __unwrapped_reverse_subrange_t<remove_cvref_t<_Range>>)
  _LIBCUDACXX_REQUIRES(__is_unsized_reverse_subrange<remove_cvref_t<_Range>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(_UnwrappedSubrange(__range.end().base(), __range.begin().base()))) -> _UnwrappedSubrange
  {
    return _UnwrappedSubrange(__range.end().base(), __range.begin().base());
  }

  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(
    (!__is_reverse_view<remove_cvref_t<_Range>>) _LIBCUDACXX_AND(!__is_sized_reverse_subrange<remove_cvref_t<_Range>>)
      _LIBCUDACXX_AND(!__is_unsized_reverse_subrange<remove_cvref_t<_Range>>))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(reverse_view{_CUDA_VSTD::forward<_Range>(__range)})) -> reverse_view<all_t<_Range>>
  {
    return reverse_view{_CUDA_VSTD::forward<_Range>(__range)};
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto reverse = __reverse::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___RANGES_REVERSE_VIEW_H
