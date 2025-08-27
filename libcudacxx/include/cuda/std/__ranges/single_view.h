// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA_STD___RANGES_SINGLE_VIEW_H
#define _CUDA_STD___RANGES_SINGLE_VIEW_H

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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <move_constructible _Tp>
  requires is_object_v<_Tp>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp, enable_if_t<move_constructible<_Tp>, int> = 0, enable_if_t<is_object_v<_Tp>, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class single_view : public view_interface<single_view<_Tp>>
{
  _CCCL_NO_UNIQUE_ADDRESS __movable_box<_Tp> __value_;

public:
#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI single_view()
    requires default_initializable<_Tp>
  = default;
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(default_initializable<_Tp2>)
  _CCCL_API constexpr single_view() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>()
      , __value_(){};
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

  _CCCL_TEMPLATE(class _Tp2 = _Tp) // avoids circular concept definitions with copy_constructible
  _CCCL_REQUIRES((!is_same_v<remove_cvref_t<_Tp2>, single_view>) _CCCL_AND copy_constructible<_Tp2>)
  _CCCL_API constexpr explicit single_view(const _Tp2& __t) noexcept(is_nothrow_copy_constructible_v<_Tp2>)
      : view_interface<single_view<_Tp2>>()
      , __value_(in_place, __t)
  {}

  _CCCL_API constexpr explicit single_view(_Tp&& __t) noexcept(is_nothrow_move_constructible_v<_Tp>)
      : view_interface<single_view<_Tp>>()
      , __value_(in_place, ::cuda::std::move(__t))
  {}

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(constructible_from<_Tp, _Args...>)
  _CCCL_API constexpr explicit single_view(in_place_t,
                                           _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : view_interface<single_view<_Tp>>()
      , __value_{in_place, ::cuda::std::forward<_Args>(__args)...}
  {}

  [[nodiscard]] _CCCL_API constexpr _Tp* begin() noexcept
  {
    return data();
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* begin() const noexcept
  {
    return data();
  }

  [[nodiscard]] _CCCL_API constexpr _Tp* end() noexcept
  {
    return data() + 1;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* end() const noexcept
  {
    return data() + 1;
  }

  [[nodiscard]] _CCCL_API static constexpr size_t size() noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_API constexpr _Tp* data() noexcept
  {
    return __value_.operator->();
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* data() const noexcept
  {
    return __value_.operator->();
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE single_view(_Tp) -> single_view<_Tp>;

_CCCL_END_NAMESPACE_RANGES

_CCCL_BEGIN_NAMESPACE_VIEWS
_CCCL_BEGIN_NAMESPACE_CPO(__single_view)

template <class _Tp>
_CCCL_CONCEPT __can_single_view = _CCCL_REQUIRES_EXPR((_Tp))(typename(single_view<decay_t<_Tp>>));

struct __fn : __range_adaptor_closure<__fn>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_single_view<_Tp>) // MSVC breaks without it
  _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(single_view<decay_t<_Tp>>(::cuda::std::forward<_Tp>(__t)))) -> single_view<decay_t<_Tp>>
  {
    return single_view<decay_t<_Tp>>(::cuda::std::forward<_Tp>(__t));
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto single = __single_view::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_VIEWS

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_SINGLE_VIEW_H
