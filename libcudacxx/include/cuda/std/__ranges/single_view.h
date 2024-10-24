// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_SINGLE_VIEW_H
#define _LIBCUDACXX___RANGES_SINGLE_VIEW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__ranges/movable_box.h>
#include <cuda/std/__ranges/range_adaptor.h>
#include <cuda/std/__ranges/view_interface.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

#  if _CCCL_STD_VER >= 2020
template <copy_constructible _Tp>
  requires is_object_v<_Tp>
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp, enable_if_t<copy_constructible<_Tp>, int> = 0, enable_if_t<is_object_v<_Tp>, int> = 0>
#  endif // _CCCL_STD_VER <= 2017
class single_view : public view_interface<single_view<_Tp>>
{
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Tp> __value_;

public:
#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI single_view()
    requires default_initializable<_Tp>
  = default;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES(default_initializable<_Tp2>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr single_view() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>()
      , __value_(){};
#  endif // _CCCL_STD_VER <= 2017

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit single_view(const _Tp& __t) noexcept(is_nothrow_copy_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>()
      , __value_(in_place, __t)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit single_view(_Tp&& __t) noexcept(is_nothrow_move_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>()
      , __value_(in_place, _CUDA_VSTD::move(__t))
  {}

  _LIBCUDACXX_TEMPLATE(class... _Args)
  _LIBCUDACXX_REQUIRES(constructible_from<_Tp, _Args...>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit single_view(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : view_interface<single_view<_Tp>>()
      , __value_{in_place, _CUDA_VSTD::forward<_Args>(__args)...}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* begin() noexcept
  {
    return data();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp* begin() const noexcept
  {
    return data();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* end() noexcept
  {
    return data() + 1;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp* end() const noexcept
  {
    return data() + 1;
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t size() noexcept
  {
    return 1;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* data() noexcept
  {
    return __value_.operator->();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp* data() const noexcept
  {
    return __value_.operator->();
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE single_view(_Tp) -> single_view<_Tp>;

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_VIEWS
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__single_view)

template <class _Range>
_LIBCUDACXX_CONCEPT_FRAGMENT(__can_single_view_, requires(_Range&& __range)(typename(single_view<decay_t<_Range>>)));

template <class _Range>
_LIBCUDACXX_CONCEPT __can_single_view = _LIBCUDACXX_FRAGMENT(__can_single_view_, _Range);

struct __fn : __range_adaptor_closure<__fn>
{
  _LIBCUDACXX_TEMPLATE(class _Range)
  _LIBCUDACXX_REQUIRES(__can_single_view<_Range>) // MSVC breaks without it
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(_Range&& __range) const noexcept(
    noexcept(single_view<decay_t<_Range>>(_CUDA_VSTD::forward<_Range>(__range)))) -> single_view<decay_t<_Range>>
  {
    return single_view<decay_t<_Range>>(_CUDA_VSTD::forward<_Range>(__range));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto single = __single_view::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_VIEWS

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___RANGES_SINGLE_VIEW_H
