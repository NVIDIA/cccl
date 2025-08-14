// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_COMMON_VIEW_H
#define _LIBCUDACXX___RANGES_COMMON_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__iterator/common_iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <view _View>
  requires(!common_range<_View> && copyable<iterator_t<_View>>)
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _View,
          enable_if_t<view<_View>, int>                 = 0,
          enable_if_t<!common_range<_View>, int>        = 0,
          enable_if_t<copyable<iterator_t<_View>>, int> = 0>
#endif // !_CCCL_HAS_CONCEPTS()
class common_view : public view_interface<common_view<_View>>
{
  _CCCL_NO_UNIQUE_ADDRESS _View __base_ = _View();

public:
#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI common_view()
    requires default_initializable<_View>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(default_initializable<_View2>)
  _CCCL_API constexpr common_view() noexcept(is_nothrow_default_constructible_v<_View2>)
      : view_interface<common_view<_View>>()
  {}
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr explicit common_view(_View __v) noexcept(is_nothrow_move_constructible_v<_View>)
      : __base_(_CUDA_VSTD::move(__v))
  {}

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(copy_constructible<_View2>)
  _CCCL_API constexpr _View base() const& noexcept(is_nothrow_copy_constructible_v<_View2>)
  {
    return __base_;
  }

  _CCCL_API constexpr _View base() && noexcept(is_nothrow_move_constructible_v<_View>)
  {
    return _CUDA_VSTD::move(__base_);
  }

  _CCCL_API constexpr auto begin()
  {
    if constexpr (random_access_range<_View> && sized_range<_View>)
    {
      return _CUDA_VRANGES::begin(__base_);
    }
    else
    {
      return common_iterator<iterator_t<_View>, sentinel_t<_View>>(_CUDA_VRANGES::begin(__base_));
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(range<const _View2>)
  _CCCL_API constexpr auto begin() const
  {
    if constexpr (random_access_range<const _View> && sized_range<const _View>)
    {
      return _CUDA_VRANGES::begin(__base_);
    }
    else
    {
      return common_iterator<iterator_t<const _View>, sentinel_t<const _View>>(_CUDA_VRANGES::begin(__base_));
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_API constexpr auto end()
  {
    if constexpr (random_access_range<_View> && sized_range<_View>)
    {
      return _CUDA_VRANGES::begin(__base_) + _CUDA_VRANGES::size(__base_);
    }
    else
    {
      return common_iterator<iterator_t<_View>, sentinel_t<_View>>(_CUDA_VRANGES::end(__base_));
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(range<const _View2>)
  _CCCL_API constexpr auto end() const
  {
    if constexpr (random_access_range<const _View> && sized_range<const _View>)
    {
      return _CUDA_VRANGES::begin(__base_) + _CUDA_VRANGES::size(__base_);
    }
    else
    {
      return common_iterator<iterator_t<const _View>, sentinel_t<const _View>>(_CUDA_VRANGES::end(__base_));
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(sized_range<_View2>)
  _CCCL_API constexpr auto size()
  {
    return _CUDA_VRANGES::size(__base_);
  }

  _CCCL_TEMPLATE(class _View2 = _View)
  _CCCL_REQUIRES(sized_range<const _View2>)
  _CCCL_API constexpr auto size() const
  {
    return _CUDA_VRANGES::size(__base_);
  }
};

template <class _Range>
_CCCL_HOST_DEVICE common_view(_Range&&) -> common_view<_CUDA_VIEWS::all_t<_Range>>;

template <class _View>
inline constexpr bool enable_borrowed_range<common_view<_View>> = enable_borrowed_range<_View>;

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__common)
struct __fn : __range_adaptor_closure<__fn>
{
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(common_range<_Range>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(_CUDA_VIEWS::all(_CUDA_VSTD::forward<_Range>(__range)))) -> all_t<_Range>
  {
    return _CUDA_VIEWS::all(_CUDA_VSTD::forward<_Range>(__range));
  }

  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES((!common_range<_Range>) )
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Range&& __range) const
    noexcept(noexcept(common_view{_CUDA_VSTD::forward<_Range>(__range)})) -> common_view<all_t<_Range>>
  {
    return common_view{_CUDA_VSTD::forward<_Range>(__range)};
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto common = __common::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_COMMON_VIEW_H
