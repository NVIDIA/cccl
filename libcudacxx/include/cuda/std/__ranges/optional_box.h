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

#ifndef _CUDA_STD___RANGES_OPTIONAL_BOX_H
#define _CUDA_STD___RANGES_OPTIONAL_BOX_H

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
#include <cuda/std/__type_traits/is_comparable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

// __optional_box is a significantly stripped down implementation of std::optional used in
// ranges where default-constructability of members is required. In some cases, we can
// completely avoid the use of an empty state; we provide a specialization of __optional_box
// that does this, see below for the details.

template <class _Tp, class = void>
class __ob_destruct_base
{
public:
  union
  {
    _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;
  };
  bool __engaged_ = false;

  _CCCL_API constexpr __ob_destruct_base() noexcept {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __ob_destruct_base(in_place_t,
                                         _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_{::cuda::std::forward<_Args>(__args)...}
      , __engaged_{true}
  {}

  _CCCL_API _CCCL_CONSTEXPR_CXX20 ~__ob_destruct_base() noexcept
  {
    __reset();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API _CCCL_CONSTEXPR_CXX20 void __reset() noexcept
  {
    if (__engaged_)
    {
      __val_.~_Tp();
      __engaged_ = false;
    }
  }
};

template <class _Tp>
class __ob_destruct_base<_Tp, enable_if_t<is_trivially_destructible_v<_Tp>>>
{
public:
  union
  {
    _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;
  };
  bool __engaged_ = false;

  _CCCL_API constexpr __ob_destruct_base() noexcept {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __ob_destruct_base(in_place_t,
                                         _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_{::cuda::std::forward<_Args>(__args)...}
      , __engaged_{true}
  {}

  _CCCL_API constexpr void __reset() noexcept
  {
    __engaged_ = false;
  }
};

template <class _Tp, class = void>
class __ob_copy_base : public __ob_destruct_base<_Tp>
{
public:
  _CCCL_HIDE_FROM_ABI constexpr __ob_copy_base() noexcept              = default;
  _CCCL_HIDE_FROM_ABI __ob_copy_base(const __ob_copy_base&)            = delete;
  _CCCL_HIDE_FROM_ABI __ob_copy_base& operator=(const __ob_copy_base&) = delete;
  _CCCL_HIDE_FROM_ABI __ob_copy_base(__ob_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __ob_copy_base& operator=(__ob_copy_base&&)      = default;
};

template <class _Tp>
class __ob_copy_base<_Tp, enable_if_t<copyable<_Tp>>> : public __ob_destruct_base<_Tp>
{
public:
  using __base = __ob_destruct_base<_Tp>;

  _CCCL_HIDE_FROM_ABI constexpr __ob_copy_base() noexcept = default;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI __ob_copy_base(const __ob_copy_base&)
    requires is_trivially_copy_constructible_v<_Tp>
  = default;
#endif // _CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr __ob_copy_base(const __ob_copy_base& __other) noexcept(is_nothrow_copy_constructible_v<_Tp>)
      : __base{}
  {
    if (__other.__engaged_)
    {
      ::cuda::std::__construct_at(::cuda::std::addressof(this->__val_), __other.__val_);
      this->__engaged_ = true;
    }
  }

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI __ob_copy_base& operator=(const __ob_copy_base&)
    requires copyable<_Tp> && is_trivially_copy_assignable_v<_Tp>
  = default;
#endif // _CCCL_HAS_CONCEPTS()

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __ob_copy_base& operator=(const __ob_copy_base& __other) noexcept(
    is_nothrow_copy_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
  {
    if (this == ::cuda::std::addressof(__other))
    {
      return *this;
    }

    if (__other.__engaged_)
    {
      *this = __other.__val_;
    }
    else if (this->__engaged_)
    {
      this->__reset();
    }
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __ob_copy_base&
  operator=(const _Tp& __val) noexcept(is_nothrow_copy_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
  {
    if (this->__engaged_)
    {
      this->__val_ = __val;
    }
    else
    {
      ::cuda::std::__construct_at(::cuda::std::addressof(this->__val_), __val);
      this->__engaged_ = true;
    }

    return *this;
  }

  _CCCL_HIDE_FROM_ABI __ob_copy_base(__ob_copy_base&&)            = default;
  _CCCL_HIDE_FROM_ABI __ob_copy_base& operator=(__ob_copy_base&&) = default;
};

// Primary template - uses ::cuda::std::optional-like semantics and introduces an empty state
// in case assignment fails.
#if _CCCL_HAS_CONCEPTS()
template <movable _Tp>
class __optional_box : public __ob_copy_base<_Tp>
{
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp, class = void, class = void>
class __optional_box;

template <class _Tp>
class __optional_box<_Tp, enable_if_t<movable<_Tp>>, enable_if_t<!default_initializable<_Tp>>>
    : public __ob_copy_base<_Tp>
{
#endif // !_CCCL_HAS_CONCEPTS()

public:
  using __base = __ob_copy_base<_Tp>;

  _CCCL_HIDE_FROM_ABI constexpr __optional_box() noexcept              = default;
  _CCCL_HIDE_FROM_ABI __optional_box(const __optional_box&)            = default;
  _CCCL_HIDE_FROM_ABI __optional_box& operator=(const __optional_box&) = default;

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI __optional_box(__optional_box&&)
    requires is_trivially_move_constructible_v<_Tp>
  = default;
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr __optional_box(__optional_box&& __other) noexcept(is_nothrow_move_constructible_v<_Tp>)
      : __base{}
  {
    if (__other.__engaged_)
    {
      ::cuda::std::__construct_at(::cuda::std::addressof(this->__val_), ::cuda::std::move(__other.__val_));
      this->__engaged_ = true;
    }
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!same_as<_Up, _Tp>) _CCCL_AND convertible_to<const _Up&, _Tp>)
  _CCCL_API constexpr __optional_box(const __optional_box<_Up>& __other) noexcept(
    is_nothrow_constructible_v<_Tp, const _Up&>)
      : __base{}
  {
    if (__other)
    {
      ::cuda::std::__construct_at(::cuda::std::addressof(this->__val_), __other.__val_);
      this->__engaged_ = true;
    }
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!same_as<_Up, _Tp>) _CCCL_AND convertible_to<_Up, _Tp>)
  _CCCL_API constexpr __optional_box(__optional_box<_Up>&& __other) noexcept(is_nothrow_constructible_v<_Tp, _Up&&>)
      : __base{}
  {
    if (__other)
    {
      ::cuda::std::__construct_at(::cuda::std::addressof(this->__val_), ::cuda::std::move(__other.__val_));
      this->__engaged_ = true;
    }
  }

#if _CCCL_HAS_CONCEPTS()
  _CCCL_HIDE_FROM_ABI __optional_box& operator=(__optional_box&&)
    requires movable<_Tp> && is_trivially_move_assignable_v<_Tp>
  = default;
#endif // !_CCCL_HAS_CONCEPTS()

  _CCCL_API constexpr __optional_box& operator=(__optional_box&& __other) noexcept(
    is_nothrow_move_constructible_v<_Tp> && is_nothrow_move_assignable_v<_Tp>)
  {
    if (this == ::cuda::std::addressof(__other))
    {
      return *this;
    }

    if (__other.__engaged_)
    {
      __optional_box::operator=(::cuda::std::move(__other.__val_));
    }
    else if (this->__engaged_)
    {
      this->__reset();
    }
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __optional_box&
  operator=(_Tp&& __val) noexcept(is_nothrow_move_constructible_v<_Tp> && is_nothrow_move_assignable_v<_Tp>)
  {
    if (this->__engaged_)
    {
      this->__val_ = ::cuda::std::move(__val);
    }
    else
    {
      ::cuda::std::__construct_at(::cuda::std::addressof(this->__val_), ::cuda::std::move(__val));
      this->__engaged_ = true;
    }

    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr _Tp const& operator*() const noexcept
  {
    return this->__val_;
  }
  [[nodiscard]] _CCCL_API constexpr _Tp& operator*() noexcept
  {
    return this->__val_;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* operator->() const noexcept
  {
    return ::cuda::std::addressof(this->__val_);
  }
  [[nodiscard]] _CCCL_API constexpr _Tp* operator->() noexcept
  {
    return ::cuda::std::addressof(this->__val_);
  }

  [[nodiscard]] _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return this->__engaged_;
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(equality_comparable<_Tp2>)
  [[nodiscard]] _CCCL_API constexpr bool operator==(const __optional_box& __other) const
    noexcept(__is_cpp17_nothrow_equality_comparable_v<_Tp2, _Tp2>)
  {
    return this->__engaged_ == __other.__engaged_ && (!this->__engaged_ || this->__val_ == __other.__val_);
  }
#if _CCCL_STD_VER <= 2017
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(equality_comparable<_Tp2>)
  [[nodiscard]] _CCCL_API constexpr bool operator!=(const __optional_box& __other) const
    noexcept(__is_cpp17_nothrow_equality_comparable_v<_Tp2, _Tp2>)
  {
    return this->__engaged_ != __other.__engaged_ || (this->__engaged_ && this->__val_ != __other.__val_);
  }
#endif // _CCCL_STD_VER <= 2017
};

// Simpler box, that always default constructs a value and does not need to store `__engaged`
// clang falls over its feet when trying to evaluate the assignment operator from _Tp, so move it to a base class
template <class _Tp, class = void>
struct __ob2_copy_base
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_{};
};

template <class _Tp>
struct __ob2_copy_base<_Tp, enable_if_t<copyable<_Tp>>>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_{};

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __ob2_copy_base& operator=(const _Tp& __val) noexcept(is_nothrow_copy_assignable_v<_Tp>)
  {
    __val_ = __val;
    return *this;
  }
};

#if _CCCL_HAS_CONCEPTS()
template <movable _Tp>
  requires default_initializable<_Tp>
class __optional_box<_Tp> : public __ob2_copy_base<_Tp>
{
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
class __optional_box<_Tp, enable_if_t<movable<_Tp>>, enable_if_t<default_initializable<_Tp>>>
    : public __ob2_copy_base<_Tp>
{
#endif // !_CCCL_HAS_CONCEPTS()

public:
  using __base = __ob2_copy_base<_Tp>;

  _CCCL_API constexpr __optional_box() noexcept
      : __base{}
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!same_as<_Up, _Tp>) _CCCL_AND convertible_to<const _Up&, _Tp>)
  _CCCL_API constexpr __optional_box(const __optional_box<_Up>& __other) noexcept(
    is_nothrow_constructible_v<_Tp, const _Up&>)
  {
    if (__other)
    {
      this->__val_ = __other.__val_;
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!same_as<_Up, _Tp>) _CCCL_AND convertible_to<_Up, _Tp>)
  _CCCL_API constexpr __optional_box(__optional_box<_Up>&& __other) noexcept(is_nothrow_assignable_v<_Tp, _Up&&>)
  {
    if (__other)
    {
      this->__val_ = ::cuda::std::move(__other.__val_);
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API _CCCL_CONSTEXPR_CXX20 __optional_box& operator=(_Tp&& __val) noexcept(is_nothrow_move_assignable_v<_Tp>)
  {
    this->__val_ = ::cuda::std::move(__val);
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr _Tp const& operator*() const noexcept
  {
    return this->__val_;
  }
  [[nodiscard]] _CCCL_API constexpr _Tp& operator*() noexcept
  {
    return this->__val_;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* operator->() const noexcept
  {
    return ::cuda::std::addressof(this->__val_);
  }
  [[nodiscard]] _CCCL_API constexpr _Tp* operator->() noexcept
  {
    return ::cuda::std::addressof(this->__val_);
  }

  [[nodiscard]] _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return true;
  }

  _CCCL_API constexpr void
  __reset() noexcept(is_nothrow_default_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>)
  {
    this->__val_ = _Tp{};
  }

  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(equality_comparable<_Tp2>)
  [[nodiscard]] _CCCL_API constexpr bool operator==(const __optional_box& __other) const
    noexcept(__is_cpp17_nothrow_equality_comparable_v<_Tp2, _Tp2>)
  {
    return this->__val_ == __other.__val_;
  }
#if _CCCL_STD_VER <= 2017
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(equality_comparable<_Tp2>)
  [[nodiscard]] _CCCL_API constexpr bool operator!=(const __optional_box& __other) const
    noexcept(__is_cpp17_nothrow_equality_comparable_v<_Tp2, _Tp2>)
  {
    return this->__val_ != __other.__val_;
  }
#endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_OPTIONAL_BOX_H
