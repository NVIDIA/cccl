//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief SCOPE guards and exception handling (`on_throw`)
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__stf/utility/source_location.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace cuda::experimental::stf
{
/**
 * @brief Creates a policy that suppresses exceptions.
 *
 * Apply the policy to a callable with `on_throw(::std::ignore) << callable`.
 * If the callable throws, a `void` result is simply suppressed and a non-void
 * result is replaced with a default-constructed value. Consequently, a
 * non-void callable must return a default-constructible type.
 *
 * @return A policy object consumed by `operator<<`.
 */
inline auto on_throw(decltype(::std::ignore)) noexcept
{
  struct __result
  {};
  return __result{};
}

template <class _Fn>
decltype(auto) operator<<(decltype(on_throw(::std::ignore)), _Fn&& __fn) noexcept
{
  using _Result = decltype(::cuda::std::forward<_Fn>(__fn)());
  static_assert(::cuda::std::is_void_v<_Result> || ::cuda::std::is_default_constructible_v<_Result>,
                "on_throw(std::ignore) requires a default-constructible result");
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__fn)();
  }
  _CCCL_CATCH_ALL
  {
    if constexpr (!::cuda::std::is_void_v<_Result>)
    {
      return _Result{};
    }
  }
}

/**
 * @brief The type of a handler that reports an exception and lets execution resume.
 *
 * The `decltype(::std::ignore)` return type marks the handler as suppressing; the returned
 * value itself is discarded. The handler receives a pointer to the caught `std::exception`,
 * or `nullptr` if the exception does not derive from `std::exception`, together with the
 * location captured at the `on_throw` call site.
 */
using throw_handler_ignore_t =
  decltype(::std::ignore) (*)(const ::std::exception*, ::cuda::std::source_location) noexcept;

/**
 * @brief The type of a handler that reports an exception and is expected not to return.
 *
 * The `void` return type marks the handler as terminating. `[[noreturn]]` appertains to a
 * function declaration rather than to a function type, so it cannot be part of this alias
 * and the contract is unenforceable; `::std::abort` runs if the handler returns anyway. The
 * handler receives a pointer to the caught `std::exception`, or `nullptr` if the exception
 * does not derive from `std::exception`, together with the location captured at the
 * `on_throw` call site.
 */
using throw_handler_terminate_t = void (*)(const ::std::exception*, ::cuda::std::source_location) noexcept;

/**
 * @brief Creates a policy that hands an exception to a handler and then suppresses it.
 *
 * Apply the policy with `on_throw(handler) << callable`. If the callable throws, the handler
 * is invoked with the caught exception (or `nullptr` for an exception that does not derive
 * from `std::exception`) and `__loc`. Execution then resumes: a `void` result is suppressed
 * and a non-void result is replaced with a default-constructed value. Consequently, a
 * non-void callable must return a default-constructible type.
 *
 * @param[in] __handler A non-null suppressing handler.
 * @param[in] __loc The location passed to the handler; defaults to the call site.
 * @return A policy object consumed by `operator<<`.
 */
inline auto on_throw(throw_handler_ignore_t __handler,
                     const ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  struct __result
  {
    throw_handler_ignore_t __handler_;
    ::cuda::std::source_location __loc_;
  };
  return __result{__handler, __loc};
}

template <class _Fn>
decltype(auto) operator<<(decltype(on_throw(throw_handler_ignore_t{})) __policy, _Fn&& __fn) noexcept
{
  using _Result = decltype(::cuda::std::forward<_Fn>(__fn)());
  static_assert(::cuda::std::is_void_v<_Result> || ::cuda::std::is_default_constructible_v<_Result>,
                "on_throw(throw_handler_ignore_t) requires a default-constructible result");
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__fn)();
  }
  _CCCL_CATCH (const ::std::exception& __exception)
  {
    __policy.__handler_(&__exception, __policy.__loc_);
  }
  _CCCL_CATCH_ALL
  {
    __policy.__handler_(nullptr, __policy.__loc_);
  }

  if constexpr (!::cuda::std::is_void_v<_Result>)
  {
    return _Result{};
  }
}

/**
 * @brief Creates a policy that hands an exception to a handler that must not return.
 *
 * Apply the policy with `on_throw(handler) << callable`. If the callable throws, the handler
 * is invoked with the caught exception (or `nullptr` for an exception that does not derive
 * from `std::exception`) and `__loc`. The handler is assumed to terminate; `::std::abort`
 * runs immediately afterwards in case it returns.
 *
 * @param[in] __handler A non-null terminating handler.
 * @param[in] __loc The location passed to the handler; defaults to the call site.
 * @return A policy object consumed by `operator<<`.
 */
inline auto on_throw(throw_handler_terminate_t __handler,
                     const ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  struct __result
  {
    throw_handler_terminate_t __handler_;
    ::cuda::std::source_location __loc_;
  };
  return __result{__handler, __loc};
}

template <class _Fn>
decltype(auto) operator<<(decltype(on_throw(throw_handler_terminate_t{})) __policy, _Fn&& __fn) noexcept
{
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__fn)();
  }
  _CCCL_CATCH (const ::std::exception& __exception)
  {
    __policy.__handler_(&__exception, __policy.__loc_);
  }
  _CCCL_CATCH_ALL
  {
    __policy.__handler_(nullptr, __policy.__loc_);
  }
  ::std::abort();
  _CCCL_UNREACHABLE();
}

/**
 * @brief Creates a policy that reports and suppresses exceptions.
 *
 * Apply the policy with `on_throw(stream) << callable`. If the callable throws,
 * the policy writes the captured call-site location and exception message to
 * `stream`, flushes it, and returns a default-constructed result for a non-void
 * callable. Non-standard exceptions are reported without a message.
 *
 * @param[in] __stream A non-null writable C stream.
 * @param[in] __loc The location reported on failure; defaults to the call site.
 * @return A policy object consumed by `operator<<`.
 */
inline auto on_throw(::FILE* __stream,
                     const ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  struct __result
  {
    ::FILE* __stream_;
    ::cuda::std::source_location __loc_;
  };
  return __result{__stream, __loc};
}

template <class _Fn>
decltype(auto) operator<<(decltype(on_throw(stderr)) __policy, _Fn&& __fn) noexcept
{
  using _Result = decltype(::cuda::std::forward<_Fn>(__fn)());
  static_assert(::cuda::std::is_void_v<_Result> || ::cuda::std::is_default_constructible_v<_Result>,
                "on_throw(FILE*) requires a default-constructible result");
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__fn)();
  }
  _CCCL_CATCH (const ::std::exception& __exception)
  {
    ::fprintf(__policy.__stream_,
              "%s(%u) on_throw violation in %s: %s\n",
              __policy.__loc_.file_name(),
              __policy.__loc_.line(),
              __policy.__loc_.function_name(),
              __exception.what());
  }
  _CCCL_CATCH_ALL
  {
    ::fprintf(__policy.__stream_,
              "%s(%u) on_throw violation in %s: nonstandard exception\n",
              __policy.__loc_.file_name(),
              __policy.__loc_.line(),
              __policy.__loc_.function_name());
  }
  ::fflush(__policy.__stream_);

  if constexpr (!::cuda::std::is_void_v<_Result>)
  {
    return _Result{};
  }
}

/**
 * @brief Creates a policy that reports an exception and terminates.
 *
 * Apply the policy with `on_throw(::std::abort) << callable` or
 * `on_throw(::std::terminate) << callable`. If the callable throws, the policy
 * reports the exception to `stderr`, flushes the stream, and invokes the chosen
 * handler. Passing any other function pointer reports the invalid handler and
 * terminates immediately.
 *
 * @param[in] __handler The address of `std::abort` or `std::terminate`.
 * @param[in] __loc The location reported on failure; defaults to the call site.
 * @return A policy object consumed by `operator<<`.
 */
inline auto on_throw(void (*__handler)() noexcept,
                     const ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  // The function-pointer type also accepts any other `void() noexcept` function,
  // so overload resolution alone cannot restrict this argument to these two values.
  if (__handler != &::std::abort && __handler != &::std::terminate)
  {
    ::fprintf(stderr,
              "%s(%u) invalid on_throw termination handler in %s; expected std::abort or std::terminate\n",
              __loc.file_name(),
              __loc.line(),
              __loc.function_name());
    ::fflush(stderr);
    ::std::terminate();
  }
  struct __result
  {
    void (*__handler_)() noexcept;
    ::cuda::std::source_location __loc_;
  };
  return __result{__handler, __loc};
}

template <class _Fn>
decltype(auto) operator<<(decltype(on_throw(::std::abort)) __policy, _Fn&& __fn) noexcept
{
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__fn)();
  }
  _CCCL_CATCH_ALL
  {
    on_throw(stderr, __policy.__loc_) << [] {
      throw;
    };
  }
  __policy.__handler_();
  _CCCL_UNREACHABLE();
}

#ifdef UNITTESTED_FILE
UNITTEST("on_throw")
{
  using namespace cuda::experimental::stf;
  //! [on_throw]
  int value = 0;
  on_throw(::std::abort) << [&] {
    value = 42; // would abort the application if this code threw
  };
  on_throw(::std::terminate) << [] {};
  on_throw(stderr) << [] {};
  EXPECT(value == 42);
  //! [on_throw]

  // A terminating handler stays out of the way as long as nothing throws.
  const throw_handler_terminate_t die = [](const ::std::exception*, ::cuda::std::source_location) noexcept {
    ::std::abort();
  };
  const int untouched = on_throw(die) << [] {
    return 7;
  };
  EXPECT(untouched == 7);

#  if _CCCL_HAS_EXCEPTIONS()
  const int ignored = on_throw(::std::ignore) << []() -> int {
    throw ::std::runtime_error("ignored");
  };
  EXPECT(ignored == 0);
  on_throw(::std::ignore) << [] {
    throw 42;
  };

  ::FILE* const log = ::tmpfile();
  EXPECT(log != nullptr);
  const auto loc   = ::cuda::std::source_location::current();
  const int logged = on_throw(log, loc) << []() -> int {
    throw ::std::runtime_error("boom");
  };
  EXPECT(logged == 0);
  ::rewind(log);
  char message[1024]{};
  EXPECT(::fgets(message, sizeof(message), log) != nullptr);
  char expected[1024]{};
  ::snprintf(expected,
             sizeof(expected),
             "%s(%u) on_throw violation in %s: boom\n",
             loc.file_name(),
             loc.line(),
             loc.function_name());
  EXPECT(::std::string_view{message} == expected);
  ::fclose(log);

  // A suppressing handler sees the exception and the call site, then execution resumes.
  static bool saw_std_exception = false;
  static unsigned reported_line = 0;
  const throw_handler_ignore_t note =
    [](const ::std::exception* __exception, ::cuda::std::source_location __where) noexcept -> decltype(::std::ignore) {
    saw_std_exception = __exception != nullptr;
    reported_line     = __where.line();
    return ::std::ignore;
  };

  const auto site = ::cuda::std::source_location::current();
  const int noted = on_throw(note, site) << []() -> int {
    throw ::std::runtime_error("noted");
  };
  EXPECT(noted == 0);
  EXPECT(saw_std_exception);
  EXPECT(reported_line == site.line());

  // An exception that does not derive from std::exception reaches the handler as nullptr.
  on_throw(note, site) << [] {
    throw 42;
  };
  EXPECT(!saw_std_exception);
#  endif // _CCCL_HAS_EXCEPTIONS()
};
#endif // UNITTESTED_FILE

/**
 * @brief Automatically runs code when a scope is exited (`SCOPE(exit)`), exited by means of an exception
 * (`SCOPE(fail)`), or exited normally (`SCOPE(success)`).
 *
 * The code controlled by `SCOPE(exit)` and `SCOPE(fail)` must not throw. In debug builds (`NDEBUG` not
 * defined) those lambdas are invoked via `on_throw(::std::abort)`; in release
 * builds they are called directly. The code controlled by `SCOPE(success)` may throw. In all cases the
 * controlled code must return `void` (enforced at compile time).
 *
 * `SCOPE(exit)` runs its code at the natural termination of the current scope. Example: @snippet this SCOPE(exit)
 *
 * `SCOPE(fail)` runs its code if and only if the current scope is left by means of throwing an exception. Example:
 * @snippet this SCOPE(fail)
 *
 * Finally, `SCOPE(success)` runs its code if and only if the current scope is left by normal flow (as opposed to by an
 * exception). Example: @snippet this SCOPE(success)
 *
 * If two or more `SCOPE` declarations are present in the same scope, they will take effect in the reverse order of
 * their lexical order. Example: @snippet this SCOPE combinations
 *
 *  See Also: https://en.cppreference.com/w/cpp/experimental/scope_exit,
 * https://en.cppreference.com/w/cpp/experimental/scope_fail,
 * https://en.cppreference.com/w/cpp/experimental/scope_success
 */
///@{
#define SCOPE(kind) \
  auto CUDASTF_UNIQUE_NAME(scope_guard) = (::cuda::experimental::stf::detail::scope_guard_handler::kind) {}->*[&]()
///@}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail::scope_guard_handler
{
enum class exit
{
};
enum class fail
{
};
enum class success
{
};

template <class F>
void invoke_nothrow(F& f, ::cuda::std::source_location loc)
{
  static_assert(::std::is_void_v<decltype(f())>, "SCOPE requires a void-returning callable");
#  ifndef NDEBUG
  on_throw(::std::abort, loc) << f;
#  else // ^^^ !NDEBUG ^^^ / vvv NDEBUG vvv
  (void) loc;
  f();
#  endif // NDEBUG
}

template <typename F>
auto operator->*(with_location<exit> where, F&& f)
{
  struct result
  {
    F f;
    const ::cuda::std::source_location loc;
    // Armed when != -1; move sets -1 to disarm. Value is otherwise unused for exit.
    int exceptions = 0;

    result(F&& f, ::cuda::std::source_location loc)
        : f(::std::forward<F>(f))
        , loc(loc)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , loc(rhs.loc)
        , exceptions(::std::exchange(rhs.exceptions, -1))
    {}

    ~result() noexcept
    {
      if (exceptions != -1)
      {
        invoke_nothrow(f, loc);
      }
    }
  };

  return result{::std::forward<F>(f), where.loc};
}

template <typename F>
auto operator->*(with_location<fail> where, F&& f)
{
  struct result
  {
    F f;
    const ::cuda::std::source_location loc;
    // Expected uncaught count, or -1 when disarmed by move.
    int exceptions;

    result(F&& f, ::cuda::std::source_location loc, int exceptions)
        : f(::std::forward<F>(f))
        , loc(loc)
        , exceptions(exceptions)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , loc(rhs.loc)
        , exceptions(::std::exchange(rhs.exceptions, -1))
    {}

    ~result() noexcept
    {
      if (::std::uncaught_exceptions() == exceptions)
      {
        invoke_nothrow(f, loc);
      }
    }
  };

  // Run only if an exception is in flight: uncaught count is one above creation-time count.
  return result{::std::forward<F>(f), where.loc, ::std::uncaught_exceptions() + 1};
}

template <typename F>
auto operator->*(success, F&& f)
{
  // success may throw, so it does not go through invoke_nothrow; keep the same void check.
  static_assert(::std::is_void_v<decltype(::std::forward<F>(f)())>, "SCOPE requires a void-returning callable");

  struct result
  {
    F f;
    // Expected uncaught count, or -1 when disarmed by move.
    int exceptions;

    result(F&& f, int exceptions)
        : f(::std::forward<F>(f))
        , exceptions(exceptions)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , exceptions(::std::exchange(rhs.exceptions, -1))
    {}

    // May throw — unlike exit/fail.
    ~result() noexcept(false)
    {
      if (::std::uncaught_exceptions() == exceptions)
      {
        f();
      }
    }
  };

  return result{::std::forward<F>(f), ::std::uncaught_exceptions()};
}
} // namespace detail::scope_guard_handler
#endif // !_CCCL_DOXYGEN_INVOKED
} // namespace cuda::experimental::stf

#ifdef UNITTESTED_FILE
UNITTEST("SCOPE(exit)")
{
  //! [SCOPE(exit)]
  // SCOPE(exit) runs the lambda upon the termination of the current scope.
  bool done = false;
  {
    SCOPE(exit)
    {
      done = true;
    };
    EXPECT(!done, "SCOPE_EXIT should not run early.");
  }
  EXPECT(done);
  //! [SCOPE(exit)]
};

UNITTEST("SCOPE(fail)")
{
  //! [SCOPE(fail)]
  bool done = false;
  {
    SCOPE(fail)
    {
      done = true;
    };
    EXPECT(!done, "SCOPE_FAIL should not run early.");
  }
  EXPECT(!done);

  try
  {
    SCOPE(fail)
    {
      done = true;
    };
    EXPECT(!done);
    throw 42;
  }
  catch (...)
  {
    EXPECT(done);
  }
  //! [SCOPE(fail)]
};

UNITTEST("SCOPE(success)")
{
  //! [SCOPE(success)]
  bool done = false;
  {
    SCOPE(success)
    {
      done = true;
    };
    EXPECT(!done);
  }
  EXPECT(done);
  done = false;

  try
  {
    SCOPE(success)
    {
      done = true;
    };
    EXPECT(!done);
    throw 42;
  }
  catch (...)
  {
    EXPECT(!done);
  }
  //! [SCOPE(success)]
};

UNITTEST("SCOPE combinations")
{
  //! [SCOPE combinations]
  int counter = 0;
  {
    SCOPE(exit)
    {
      EXPECT(counter == 2);
      counter = 0;
    };
    SCOPE(success)
    {
      EXPECT(counter == 1);
      ++counter;
    };
    SCOPE(exit)
    {
      EXPECT(counter == 0);
      ++counter;
    };
    EXPECT(counter == 0);
  }
  EXPECT(counter == 0);
  //! [SCOPE combinations]
};

#endif // UNITTESTED_FILE
