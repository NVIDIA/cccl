// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_NON_PROPAGATING_CACHE_H
#define _LIBCUDACXX___RANGES_NON_PROPAGATING_CACHE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h> // indirectly_readable
#include <cuda/std/__iterator/iterator_traits.h> // iter_reference_t
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

struct __forward_tag
{};
struct __from_tag
{};
// This helper class is needed to perform copy and move elision when constructing the contained type from an iterator.
template <class _Tp>
struct __wrapper
{
  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __wrapper(__forward_tag, _Args&&... __args)
      : __t_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  template <class _Fn>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __wrapper(__from_tag, const _Fn& __fun)
      : __t_(__fun())
  {}

  _Tp __t_;
};

template <class _Tp, class = void>
struct __npc_destruct_base
{
  union
  {
    _CCCL_NO_UNIQUE_ADDRESS __wrapper<_Tp> __val_;
  };
  bool __engaged_ = false;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __npc_destruct_base() noexcept {}

  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __npc_destruct_base(__forward_tag, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(__forward_tag{}, _CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 ~__npc_destruct_base() noexcept
  {
    if (__engaged_)
    {
      __val_.~_Tp();
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 void __reset() noexcept
  {
    if (__engaged_)
    {
      __val_.~_Tp();
      __engaged_ = false;
    }
  }
};

template <class _Tp>
struct __npc_destruct_base<_Tp, enable_if_t<is_trivially_destructible_v<_Tp>>>
{
  union
  {
    _CCCL_NO_UNIQUE_ADDRESS __wrapper<_Tp> __val_;
  };
  bool __engaged_ = false;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __npc_destruct_base() noexcept {}

  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __npc_destruct_base(__forward_tag, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(__forward_tag{}, _CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __reset() noexcept
  {
    if (__engaged_)
    {
      __engaged_ = false;
    }
  }
};

// __non_propagating_cache is a helper type that allows storing an optional value in it,
// but which does not copy the source's value when it is copy constructed/assigned to,
// and which resets the source's value when it is moved-from.
//
// This type is used as an implementation detail of some views that need to cache the
// result of `begin()` in order to provide an amortized O(1) begin() method. Typically,
// we don't want to propagate the value of the cache upon copy because the cached iterator
// may refer to internal details of the source view.
template <class _Tp, enable_if_t<is_object_v<_Tp>, int> = 0>
class _CCCL_TYPE_VISIBILITY_DEFAULT __non_propagating_cache : public __npc_destruct_base<_Tp>
{
public:
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__non_propagating_cache, __npc_destruct_base, _Tp);

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __non_propagating_cache(const __non_propagating_cache&) noexcept
      : __base()
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __non_propagating_cache(__non_propagating_cache&& __other) noexcept
      : __base()
  {
    __other.__reset();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __non_propagating_cache& operator=(const __non_propagating_cache& __other) noexcept
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      this->__reset();
    }
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __non_propagating_cache& operator=(__non_propagating_cache&& __other) noexcept
  {
    this->__reset();
    __other.__reset();
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp& operator*() noexcept
  {
    return this->__val_.__t_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp const& operator*() const noexcept
  {
    return this->__val_.__t_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __has_value() const noexcept
  {
    return this->__engaged_;
  }

  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp&
  __emplace(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
  {
    _CUDA_VSTD::__construct_at(
      _CUDA_VSTD::addressof(this->__val_), __forward_tag{}, _CUDA_VSTD::forward<_Args>(__args)...);
    this->__engaged_ = true;
    return this->__val_.__t_;
  }

  template <class _Fn>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp&
  __emplace_from(const _Fn& __fun) noexcept(is_nothrow_constructible_v<_Tp, invoke_result_t<_Fn>>)
  {
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __from_tag{}, __fun);
    this->__engaged_ = true;
    return this->__val_.__t_;
  }
};

struct __empty_cache
{};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

#endif // _LIBCUDACXX___RANGES_NON_PROPAGATING_CACHE_H
