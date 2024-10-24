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

#ifndef _LIBCUDACXX___RANGES_DEFAULTABLE_BOX_H
#define _LIBCUDACXX___RANGES_DEFAULTABLE_BOX_H

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
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

// MSVC complains about [[msvc::no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

// __defaultable_box allows turning a type that is movable (but maybe not default constructible) into
// a type that also default_constructible

// In some cases, we can completely avoid the use of an empty state; we provide a specialization of
// __defaultable_box that does this, see below for the details.

template <class _Tp, class _Up>
_CCCL_INLINE_VAR constexpr bool __is_nothrow_comparable =
  noexcept(_CUDA_VSTD::declval<const _Tp&>() == _CUDA_VSTD::declval<const _Up&>());

template <class _Tp, class = void>
struct __db_destruct_base
{
  union
  {
    _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;
  };
  bool __engaged_ = false;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_destruct_base() noexcept {};

  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_destruct_base(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 ~__db_destruct_base() noexcept
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
struct __db_destruct_base<_Tp, enable_if_t<is_trivially_destructible_v<_Tp>>>
{
  union
  {
    _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;
  };
  bool __engaged_ = false;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_destruct_base() noexcept {};

  template <class... _Args>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_destruct_base(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
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

template <class _Tp, class = void>
struct __db_copy_base : public __db_destruct_base<_Tp>
{
  using __base = __db_destruct_base<_Tp>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_copy_base() noexcept
      : __base()
  {}

  _CCCL_HIDE_FROM_ABI __db_copy_base(const __db_copy_base&)            = delete;
  _CCCL_HIDE_FROM_ABI __db_copy_base& operator=(const __db_copy_base&) = delete;

  _CCCL_HIDE_FROM_ABI __db_copy_base(__db_copy_base&&)            = default;
  _CCCL_HIDE_FROM_ABI __db_copy_base& operator=(__db_copy_base&&) = default;
};

template <class _Tp>
struct __db_copy_base<_Tp, enable_if_t<copyable<_Tp>>> : public __db_destruct_base<_Tp>
{
  using __base = __db_destruct_base<_Tp>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_copy_base() noexcept
      : __base()
  {}

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI __db_copy_base(const __db_copy_base&)
    requires is_trivially_copy_constructible_v<_Tp>
  = default;
#  endif // _CCCL_STD_VER >= 2020

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_copy_base(const __db_copy_base& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>)
      : __base()
  {
    if (__other.__engaged_)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
      this->__engaged_ = true;
    }
  }

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI __db_copy_base& operator=(const __db_copy_base&)
    requires copyable<_Tp> && is_trivially_copy_assignable_v<_Tp>
  = default;
#  endif

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_copy_base& operator=(const __db_copy_base& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    if (this->__engaged_)
    {
      if (__other.__engaged_)
      {
        this->__val_ = __other.__val_;
      }
      else
      {
        if constexpr (!is_trivially_destructible_v<_Tp>)
        {
          this->__val_.~_Tp();
        }
        this->__engaged_ = false;
      }
    }
    else
    {
      if (__other.__engaged_)
      {
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
        this->__engaged_ = true;
      }
      else
      {
        /* nothing to do */
      }
    }
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db_copy_base&
  operator=(const _Tp& __val) noexcept(is_nothrow_copy_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
  {
    if (this->__engaged_)
    {
      this->__val_ = __val;
    }
    else
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __val);
      this->__engaged_ = true;
    }

    return *this;
  }

  _CCCL_HIDE_FROM_ABI __db_copy_base(__db_copy_base&&)            = default;
  _CCCL_HIDE_FROM_ABI __db_copy_base& operator=(__db_copy_base&&) = default;
};

// Primary template - uses _CUDA_VSTD::optional and introduces an empty state in case assignment fails.
#  if _CCCL_STD_VER >= 2020
template <movable _Tp>
class __defaultable_box : public __db_copy_base<_Tp>
{
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp, class = void, class = void>
class __defaultable_box;

template <class _Tp>
class __defaultable_box<_Tp, enable_if_t<movable<_Tp>>, enable_if_t<!default_initializable<_Tp>>>
    : public __db_copy_base<_Tp>
{
#  endif // _CCCL_STD_VER <= 2017

public:
  using __base = __db_copy_base<_Tp>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box() noexcept
      : __base()
  {}

  _CCCL_HIDE_FROM_ABI __defaultable_box(const __defaultable_box&)            = default;
  _CCCL_HIDE_FROM_ABI __defaultable_box& operator=(const __defaultable_box&) = default;

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI __defaultable_box(__defaultable_box&&)
    requires is_trivially_move_constructible_v<_Tp>
  = default;
#  endif // _CCCL_STD_VER >= 2020

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box(__defaultable_box&& __other) noexcept(
    is_nothrow_move_constructible_v<_Tp>)
      : __base()
  {
    if (__other.__engaged_)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
      this->__engaged_ = true;
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up)
  _LIBCUDACXX_REQUIRES((!same_as<_Up, _Tp>) _LIBCUDACXX_AND convertible_to<const _Up&, _Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box(const __defaultable_box<_Up>& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>)
      : __base()
  {
    if (__other)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), __other.__val_);
      this->__engaged_ = true;
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up)
  _LIBCUDACXX_REQUIRES((!same_as<_Up, _Tp>) _LIBCUDACXX_AND convertible_to<_Up, _Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box(__defaultable_box<_Up>&& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>)
      : __base()
  {
    if (__other)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
      this->__engaged_ = true;
    }
  }

#  if _CCCL_STD_VER >= 2020
  _CCCL_HIDE_FROM_ABI __defaultable_box& operator=(__defaultable_box&&)
    requires movable<_Tp> && is_trivially_move_assignable_v<_Tp>
  = default;
#  endif // _CCCL_STD_VER >= 2020

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box& operator=(__defaultable_box&& __other) noexcept(
    is_nothrow_move_constructible_v<_Tp> && is_nothrow_move_assignable_v<_Tp>)
  {
    if (this == _CUDA_VSTD::addressof(__other))
    {
      return *this;
    }

    if (this->__engaged_)
    {
      if (__other.__engaged_)
      {
        this->__val_ = _CUDA_VSTD::move(__other.__val_);
      }
      else
      {
        this->__val_.~_Tp();
        this->__engaged_ = false;
      }
    }
    else
    {
      if (__other.__engaged_)
      {
        _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__other.__val_));
        this->__engaged_ = true;
      }
      else
      {
        /* nothing to do */
      }
    }
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box&
  operator=(_Tp&& __val) noexcept(is_nothrow_move_constructible_v<_Tp> && is_nothrow_move_assignable_v<_Tp>)
  {
    if (this->__engaged_)
    {
      this->__val_ = _CUDA_VSTD::move(__val);
    }
    else
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__val_), _CUDA_VSTD::move(__val));
      this->__engaged_ = true;
    }

    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp const& operator*() const noexcept
  {
    return this->__val_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp& operator*() noexcept
  {
    return this->__val_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp* operator->() const noexcept
  {
    return _CUDA_VSTD::addressof(this->__val_);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* operator->() noexcept
  {
    return _CUDA_VSTD::addressof(this->__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit operator bool() const noexcept
  {
    return this->__engaged_;
  }

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES(equality_comparable<_Tp2>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator==(const __defaultable_box& __other) const
    noexcept(__is_nothrow_comparable<_Tp2, _Tp2>)
  {
    return this->__engaged_ == __other.__engaged_ && (!this->__engaged_ || this->__val_ == __other.__val_);
  }
#  if _CCCL_STD_VER <= 2017
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES(equality_comparable<_Tp2>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator!=(const __defaultable_box& __other) const
    noexcept(__is_nothrow_comparable<_Tp2, _Tp2>)
  {
    return this->__engaged_ != __other.__engaged_ || (this->__engaged_ && this->__val_ != __other.__val_);
  }
#  endif // _CCCL_STD_VER <= 2017
};

// Simpler box, that always default constructs a value and does not need to store `__engaged`
// clang falls over its feet when trying to evaluate the assignment operator from _Tp, so move it to a base class
template <class _Tp, class = void>
struct __db2_copy_base
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_{};
};

template <class _Tp>
struct __db2_copy_base<_Tp, enable_if_t<copyable<_Tp>>>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_{};

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __db2_copy_base&
  operator=(const _Tp& __val) noexcept(is_nothrow_copy_assignable_v<_Tp>)
  {
    __val_ = __val;
    return *this;
  }
};
#  if _CCCL_STD_VER >= 2020
template <movable _Tp>
  requires default_initializable<_Tp>
class __defaultable_box<_Tp> : public __db2_copy_base<_Tp>
{
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Tp>
class __defaultable_box<_Tp, enable_if_t<movable<_Tp>>, enable_if_t<default_initializable<_Tp>>>
    : public __db2_copy_base<_Tp>
{
#  endif // _CCCL_STD_VER <= 2017

public:
  using __base = __db2_copy_base<_Tp>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box() noexcept
      : __base()
  {}

  _LIBCUDACXX_TEMPLATE(class _Up)
  _LIBCUDACXX_REQUIRES((!same_as<_Up, _Tp>) _LIBCUDACXX_AND convertible_to<const _Up&, _Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box(const __defaultable_box<_Up>& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>)
  {
    if (__other)
    {
      this->__val_ = __other.__val_;
    }
  }

  _LIBCUDACXX_TEMPLATE(class _Up)
  _LIBCUDACXX_REQUIRES((!same_as<_Up, _Tp>) _LIBCUDACXX_AND convertible_to<_Up, _Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __defaultable_box(__defaultable_box<_Up>&& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp>)
  {
    if (__other)
    {
      this->__val_ = _CUDA_VSTD::move(__other.__val_);
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX20 __defaultable_box&
  operator=(_Tp&& __val) noexcept(is_nothrow_move_assignable_v<_Tp>)
  {
    this->__val_ = _CUDA_VSTD::move(__val);
    return *this;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp const& operator*() const noexcept
  {
    return this->__val_;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp& operator*() noexcept
  {
    return this->__val_;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const _Tp* operator->() const noexcept
  {
    return _CUDA_VSTD::addressof(this->__val_);
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* operator->() noexcept
  {
    return _CUDA_VSTD::addressof(this->__val_);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit operator bool() const noexcept
  {
    return true;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void
  __reset() noexcept(is_nothrow_default_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
  {
    this->__val_ = _Tp{};
  }

  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES(equality_comparable<_Tp2>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator==(const __defaultable_box& __other) const
    noexcept(__is_nothrow_comparable<_Tp2, _Tp2>)
  {
    return this->__val_ == __other.__val_;
  }
#  if _CCCL_STD_VER <= 2017
  _LIBCUDACXX_TEMPLATE(class _Tp2 = _Tp)
  _LIBCUDACXX_REQUIRES(equality_comparable<_Tp2>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool operator!=(const __defaultable_box& __other) const
    noexcept(__is_nothrow_comparable<_Tp2, _Tp2>)
  {
    return this->__val_ != __other.__val_;
  }
#  endif // _CCCL_STD_VER <= 2017
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI
_LIBCUDACXX_END_NAMESPACE_RANGES

_CCCL_DIAG_POP

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___RANGES_DEFAULTABLE_BOX_H
