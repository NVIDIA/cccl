//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__EXPERIMENTAL_UTILITY_SCOPE_EXIT
#define _CUDAX__EXPERIMENTAL_UTILITY_SCOPE_EXIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/source_location>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace cuda::experimental
{
// See: https://en.cppreference.com/w/cpp/experimental/scope_exit
template <class _Fn>
struct scope_exit
{
  static_assert(::cuda::std::__is_nothrow_callable_v<_Fn&>,
                "The scope_guard function must be nothrow lvalue-callable with no arguments.");

  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2&& __fn) noexcept(::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2>)
      : scope_exit(::cuda::std::forward<_Fn2>(__fn), ::cuda::std::is_nothrow_constructible<_Fn, _Fn2>{})
  {
    static_assert(::cuda::std::is_nothrow_constructible_v<_Fn, _Fn2> || ::cuda::std::is_constructible_v<_Fn, _Fn2&>,
                  "The scope_guard function must be nothrow constructible from the provided callable or "
                  "constructible from an lvalue reference to the provided callable.");
  }

  scope_exit(scope_exit&&)            = default;
  scope_exit& operator=(scope_exit&&) = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE_API ~scope_exit()
  {
    if (__active_)
    {
      __fn_();
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE_API void release() noexcept
  {
    __active_ = false;
  }

private:
  // Handle the case where _Fn is nothrow constructible from _Fn2.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2&& __fn, ::cuda::std::true_type) noexcept
      : __fn_(::cuda::std::forward<_Fn2>(__fn))
  {}

  // Handle the case where _Fn is not nothrow constructible from _Fn2, but is
  // constructible from _Fn2&. In this case we need to make a copy of __fn first to ensure
  // that if the copy throws we don't end up with a partially constructed scope_exit
  // object. We do this by creating a temporary scope_exit object that holds a reference
  // to the original callable, and then releasing it if the copy succeeds.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2&& __fn, ::cuda::std::false_type) noexcept(false)
      : scope_exit(__fn, scope_exit<_Fn2&>(__fn))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn2>
  _CCCL_HOST_DEVICE_API explicit scope_exit(_Fn2& __fn, scope_exit<_Fn2&>&& __scope) noexcept(false)
      : __fn_(__fn) // copy not move because we don't want to invalidate __scope if the copy throws
  {
    __scope.release(); // the copy succeeded, so release __scope
  }

  _Fn __fn_;
  bool __active_{true};
};

template <class _Fn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES scope_exit(_Fn) -> scope_exit<_Fn>;

/**
 * @brief Value (or reference) plus the call-site `source_location` of its construction.
 *
 * Intended as a function parameter type: write `with_location<widget>` instead of `widget`
 * so the callee can report file/line. Construction from an argument captures
 * `source_location::current()` at that call site (overloaded operators cannot take
 * defaulted `source_location` parameters themselves).
 *
 * Move-only: may be constructed as a temporary and moved into a by-value parameter,
 * but not copied. `T` may be a value, lvalue reference, or rvalue reference.
 */
template <class _Tp>
struct with_location
{
  with_location(const with_location&)            = delete;
  with_location& operator=(const with_location&) = delete;
  with_location& operator=(with_location&&)      = delete;

  // Required so a converting temporary can initialize a by-value parameter.
  with_location(with_location&&) = default;

  // Constrained so that ill-formed reference bindings are detected by
  // `is_constructible_v` instead of erroring inside the mem-initializer, and so
  // that this template does not hijack the move constructor.
  template <typename _Up,
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::decay_t<_Up>, with_location>
                                       && ::cuda::std::is_constructible_v<_Tp, _Up&&>,
                                     int> = 0>
  constexpr with_location(_Up&& __payload, ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
      : payload(::cuda::std::forward<_Up>(__payload))
      , loc(__loc)
  {}

  _Tp payload;
  ::cuda::std::source_location loc;
};

// Two-arg form only: CTAD cannot see `_Tp` in the converting constructor, and a
// one-arg guide would steal moves (`with_location{std::move(w)}`).
template <class _Up>
with_location(_Up&&, ::cuda::std::source_location) -> with_location<::cuda::std::decay_t<_Up>>;

/**
 * @brief Invokes a callable and aborts if it throws.
 *
 * Use around code that must not let an exception escape into backend state
 * that cannot recover (e.g. after a CUDA stream capture has begun).
 *
 * Usage: `throw_proof->*[&] { ... };`
 *
 * `throw_proof` converts to `with_location`, which captures the call-site
 * `source_location` (overloaded operators cannot take default arguments). An
 * explicit `with_location{throw_proof, loc}` can forward a previously captured
 * location (CTAD, C++17).
 */
struct throw_proof_t
{
} inline constexpr throw_proof{};

template <class _Fn>
decltype(auto) operator->*(with_location<throw_proof_t> __s, _Fn&& __f) noexcept
{
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__f)();
  }
  _CCCL_CATCH (const ::std::exception& __e)
  {
    ::fprintf(stderr,
              "%s(%u) throw_proof in %s: %s\n",
              __s.loc.file_name(),
              __s.loc.line(),
              __s.loc.function_name(),
              __e.what());
  }
  _CCCL_CATCH_ALL
  {
    ::fprintf(stderr,
              "%s(%u) throw_proof in %s: unknown exception\n",
              __s.loc.file_name(),
              __s.loc.line(),
              __s.loc.function_name());
  }
  ::std::abort();
}

/**
 * @brief Invokes a callable and returns any thrown exception as an `exception_ptr`.
 *
 * Use around best-effort code where a failure should not escape (e.g. optional DOT
 * timing annotations) but the caller may still want to inspect or rethrow later.
 * The callable's return value must be `void` (enforced at compile time); the
 * result is empty if nothing was thrown.
 *
 * Usage: `auto e = throw_defer->*[&] { ... };`
 *
 * The result is `[[nodiscard]]` so the caller must acknowledge it (store it, test it,
 * or deliberately discard it, e.g. by assigning to std::ignore).
 */
struct throw_defer_t
{
} inline constexpr throw_defer{};

template <class _Fn>
[[nodiscard]] ::std::exception_ptr operator->*(throw_defer_t, _Fn&& __f) noexcept
{
  static_assert(::cuda::std::is_void_v<decltype(::cuda::std::forward<_Fn>(__f)())>,
                "throw_defer requires a void-returning callable");
  _CCCL_TRY
  {
    ::cuda::std::forward<_Fn>(__f)();
    return {};
  }
  _CCCL_CATCH_ALL
  {
    return ::std::current_exception();
  }
}
} // namespace cuda::experimental

#endif // _CUDAX__EXPERIMENTAL_UTILITY_SCOPE_EXIT
