//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_SCOPE
#define _CUDAX__UTILITY_SCOPE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_HOSTED()
#  include <exception>
#endif // _CCCL_HOSTED()

namespace cuda::experimental
{
//! @brief Scope guard that invokes its exit function when the guard is destroyed while still active.
//!
//! Models `std::experimental::scope_exit` (Library Fundamentals TS v3).
//! See: https://en.cppreference.com/w/cpp/experimental/scope_exit
template <class _Fn>
struct [[nodiscard]] scope_exit
{
  static_assert(::cuda::std::__is_nothrow_callable_v<_Fn&>,
                "scope_exit exit function must be nothrow lvalue-callable with no arguments.");

  _CCCL_TEMPLATE(class _Fn2)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Fn2>, scope_exit>)
                 && ::cuda::std::is_constructible_v<_Fn, _Fn2>)
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2&& __fn) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2> || ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2&>)
      : scope_exit(::cuda::std::forward<_Fn2>(__fn), ::cuda::std::is_nothrow_constructible<_Fn, _Fn2>{})
  {}

  // Non-template: a template would not suppress the implicit move constructor.
  _CCCL_HOST_DEVICE_API
  scope_exit(scope_exit&& __other) noexcept(::cuda::std::is_nothrow_move_constructible_v<_Fn>
                                            || ::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
      : scope_exit(::cuda::std::move(__other), ::cuda::std::is_nothrow_move_constructible<_Fn>{})
  {
    static_assert(::cuda::std::is_nothrow_move_constructible_v<_Fn> || ::cuda::std::is_copy_constructible_v<_Fn>,
                  "scope_exit move requires EF to be nothrow-move or copy constructible.");
  }

  scope_exit(const scope_exit&)            = delete;
  scope_exit& operator=(const scope_exit&) = delete;
  scope_exit& operator=(scope_exit&&)      = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE_API ~scope_exit() noexcept
  {
    if (__active_)
    {
      __fn_();
    }
  }

  _CCCL_HOST_DEVICE_API void release() noexcept
  {
    __active_ = false;
  }

private:
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2&& __fn, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::forward<_Fn2>(__fn))
  {}

  // If constructing `_Fn` from `_Fn2` may throw, first arm a guard that owns an
  // lvalue reference to `__fn`. If the copy into `__fn_` throws, that guard
  // runs `__fn()` as required by the TS; on success it is released.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2&& __fn, ::cuda::std::false_type) noexcept(false)
      : scope_exit(__fn, scope_exit<_Fn2&>(__fn))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2& __fn, scope_exit<_Fn2&>&& __guard) noexcept(false)
      : __fn_(__fn)
  {
    __guard.release();
  }

  _CCCL_HOST_DEVICE_API explicit scope_exit(scope_exit&& __other, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::move(__other.__fn_))
      , __active_(__other.__active_)
  {
    __other.release();
  }

  _CCCL_HOST_DEVICE_API explicit scope_exit(scope_exit&& __other, ::cuda::std::false_type) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
      : __fn_(__other.__fn_)
      , __active_(__other.__active_)
  {
    __other.release();
  }

  _CCCL_NO_UNIQUE_ADDRESS _Fn __fn_;
  bool __active_ = true;
};

template <class _Fn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES scope_exit(_Fn) -> scope_exit<_Fn>;

#if _CCCL_HOSTED()

//! @brief Scope guard that invokes its exit function only when destroyed during stack unwinding.
//!
//! Models `std::experimental::scope_fail` (Library Fundamentals TS v3). Host only.
//! See: https://en.cppreference.com/w/cpp/experimental/scope_fail
template <class _Fn>
struct [[nodiscard]] scope_fail
{
  static_assert(::cuda::std::__is_nothrow_callable_v<_Fn&>,
                "scope_fail exit function must be nothrow lvalue-callable with no arguments.");

  _CCCL_TEMPLATE(class _Fn2)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Fn2>, scope_fail>)
                 && ::cuda::std::is_constructible_v<_Fn, _Fn2>)
  _CCCL_HOST_API explicit scope_fail(_Fn2&& __fn) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2> || ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2&>)
      : scope_fail(::cuda::std::forward<_Fn2>(__fn), ::cuda::std::is_nothrow_constructible<_Fn, _Fn2>{})
  {}

  _CCCL_HOST_API scope_fail(scope_fail&& __other) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Fn> || ::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
      : scope_fail(::cuda::std::move(__other), ::cuda::std::is_nothrow_move_constructible<_Fn>{})
  {
    static_assert(::cuda::std::is_nothrow_move_constructible_v<_Fn> || ::cuda::std::is_copy_constructible_v<_Fn>,
                  "scope_fail move requires EF to be nothrow-move or copy constructible.");
  }

  scope_fail(const scope_fail&) = delete;

  _CCCL_HOST_API ~scope_fail() noexcept
  {
    if (__active_ && ::std::uncaught_exceptions() > __uncaught_)
    {
      __fn_();
    }
  }

  _CCCL_HOST_API void release() noexcept
  {
    __active_ = false;
  }

private:
  template <class _Fn2>
  _CCCL_HOST_API explicit scope_fail(_Fn2&& __fn, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::forward<_Fn2>(__fn))
      , __uncaught_(::std::uncaught_exceptions())
  {}

  template <class _Fn2>
  _CCCL_HOST_API explicit scope_fail(_Fn2&& __fn, ::cuda::std::false_type) noexcept(false)
      : scope_fail(__fn, scope_fail<_Fn2&>(__fn))
  {}

  template <class _Fn2>
  _CCCL_HOST_API explicit scope_fail(_Fn2& __fn, scope_fail<_Fn2&>&& __guard) noexcept(false)
      : __fn_(__fn)
      , __uncaught_(::std::uncaught_exceptions())
  {
    __guard.release();
  }

  _CCCL_HOST_API explicit scope_fail(scope_fail&& __other, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::move(__other.__fn_))
      , __uncaught_(__other.__uncaught_)
      , __active_(__other.__active_)
  {
    __other.release();
  }

  _CCCL_HOST_API explicit scope_fail(scope_fail&& __other, ::cuda::std::false_type) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
      : __fn_(__other.__fn_)
      , __uncaught_(__other.__uncaught_)
      , __active_(__other.__active_)
  {
    __other.release();
  }

  _CCCL_NO_UNIQUE_ADDRESS _Fn __fn_;
  const int __uncaught_;
  bool __active_ = true;
};

template <class _Fn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES scope_fail(_Fn) -> scope_fail<_Fn>;

//! @brief Scope guard that invokes its exit function only on normal (non-exceptional) scope exit.
//!
//! Models `std::experimental::scope_success` (Library Fundamentals TS v3). Host only.
//! The exit function may throw. See: https://en.cppreference.com/w/cpp/experimental/scope_success
template <class _Fn>
struct [[nodiscard]] scope_success
{
  static_assert(::cuda::std::__is_callable_v<_Fn&>,
                "scope_success exit function must be lvalue-callable with no arguments.");

  _CCCL_TEMPLATE(class _Fn2)
  _CCCL_REQUIRES((!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Fn2>, scope_success>)
                 && ::cuda::std::is_constructible_v<_Fn, _Fn2>)
  _CCCL_HOST_API explicit scope_success(_Fn2&& __fn) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2> || ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2&>)
      : scope_success(::cuda::std::forward<_Fn2>(__fn), ::cuda::std::is_nothrow_constructible<_Fn, _Fn2>{})
  {}

  _CCCL_HOST_API scope_success(scope_success&& __other) noexcept(
    ::cuda::std::is_nothrow_move_constructible_v<_Fn> || ::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
      : scope_success(::cuda::std::move(__other), ::cuda::std::is_nothrow_move_constructible<_Fn>{})
  {
    static_assert(::cuda::std::is_nothrow_move_constructible_v<_Fn> || ::cuda::std::is_copy_constructible_v<_Fn>,
                  "scope_success move requires EF to be nothrow-move or copy constructible.");
  }

  scope_success(const scope_success&) = delete;

  // TS: unlike exit/fail, the destructor may throw.
  _CCCL_HOST_API ~scope_success() noexcept(noexcept(::cuda::std::declval<_Fn&>()()))
  {
    if (__active_ && ::std::uncaught_exceptions() <= __uncaught_)
    {
      __fn_();
    }
  }

  _CCCL_HOST_API void release() noexcept
  {
    __active_ = false;
  }

private:
  template <class _Fn2>
  _CCCL_HOST_API explicit scope_success(_Fn2&& __fn, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::forward<_Fn2>(__fn))
      , __uncaught_(::std::uncaught_exceptions())
  {}

  // Unlike exit/fail, the TS does not require calling `fn()` if this construction throws.
  template <class _Fn2>
  _CCCL_HOST_API explicit scope_success(_Fn2&& __fn, ::cuda::std::false_type) noexcept(
    ::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2&>)
      : __fn_(__fn)
      , __uncaught_(::std::uncaught_exceptions())
  {}

  _CCCL_HOST_API explicit scope_success(scope_success&& __other, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::move(__other.__fn_))
      , __uncaught_(__other.__uncaught_)
      , __active_(__other.__active_)
  {
    __other.release();
  }

  _CCCL_HOST_API explicit scope_success(scope_success&& __other, ::cuda::std::false_type) noexcept(
    ::cuda::std::is_nothrow_copy_constructible_v<_Fn>)
      : __fn_(__other.__fn_)
      , __uncaught_(__other.__uncaught_)
      , __active_(__other.__active_)
  {
    __other.release();
  }

  _CCCL_NO_UNIQUE_ADDRESS _Fn __fn_;
  const int __uncaught_;
  bool __active_ = true;
};

template <class _Fn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES scope_success(_Fn) -> scope_success<_Fn>;
#endif // _CCCL_HOSTED()
} // namespace cuda::experimental

#endif // _CUDAX__UTILITY_SCOPE
