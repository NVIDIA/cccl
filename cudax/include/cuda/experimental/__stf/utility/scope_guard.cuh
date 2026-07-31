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
 * @brief SCOPE guards and throw sinks (`throw_proof`, `throw_defer`)
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
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__stf/utility/source_location.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>

namespace cuda::experimental::stf
{
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

template <class F>
decltype(auto) operator->*(with_location<throw_proof_t> s, F&& f) noexcept
{
  _CCCL_TRY
  {
    return ::cuda::std::forward<F>(f)();
  }
  _CCCL_CATCH (const ::std::exception& e)
  {
    ::fprintf(
      stderr, "%s(%u) throw_proof in %s: %s\n", s.loc.file_name(), s.loc.line(), s.loc.function_name(), e.what());
  }
  _CCCL_CATCH_ALL
  {
    ::fprintf(
      stderr, "%s(%u) throw_proof in %s: unknown exception\n", s.loc.file_name(), s.loc.line(), s.loc.function_name());
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

template <class F>
[[nodiscard]] ::std::exception_ptr operator->*(throw_defer_t, F&& f) noexcept
{
  static_assert(::cuda::std::is_void_v<decltype(::cuda::std::forward<F>(f)())>,
                "throw_defer requires a void-returning callable");
  _CCCL_TRY
  {
    ::cuda::std::forward<F>(f)();
    return {};
  }
  _CCCL_CATCH_ALL
  {
    return ::std::current_exception();
  }
}

/**
 * @brief Automatically runs code when a scope is exited (`SCOPE(exit)`), exited by means of an exception
 * (`SCOPE(fail)`), or exited normally (`SCOPE(success)`).
 *
 * The code controlled by `SCOPE(exit)` and `SCOPE(fail)` must not throw. In debug builds (`NDEBUG` not
 * defined) those lambdas are invoked via `throw_proof` using the `SCOPE` call-site location; in release
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
  // Forward the SCOPE call-site location into throw_proof.
  with_location{throw_proof, loc}->*f;
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
UNITTEST("throw_proof")
{
  using namespace cuda::experimental::stf;
  //! [throw_proof]
  int value = 0;
  throw_proof->*[&] {
    value = 42; // would abort the application if this code threw
  };
  EXPECT(value == 42);
  //! [throw_proof]
  EXPECT((throw_proof->*
          [] {
            return 7;
          })
         == 7);

  // Empty tag lvalues must remain convertible — that is how `throw_proof->*f`
  // captures source_location via with_location.
  static_assert(::std::is_constructible_v<with_location<throw_proof_t>, throw_proof_t&>);
  static_assert(::std::is_constructible_v<with_location<throw_proof_t>, const throw_proof_t&>);
  static_assert(::std::is_constructible_v<with_location<throw_proof_t>, throw_proof_t>);

  const auto loc = ::cuda::std::source_location::current();
  auto wl        = with_location{throw_proof, loc};
  static_assert(::std::is_same_v<decltype(wl), with_location<throw_proof_t>>);
  EXPECT(wl.loc.line() == loc.line());
};

UNITTEST("throw_defer")
{
  using namespace cuda::experimental::stf;
  //! [throw_defer]
  int value = 0;
  auto e    = throw_defer->*[&] {
    value = 42; // if this threw, e would hold the exception_ptr
  };
  EXPECT(!e);
  EXPECT(value == 42);
  //! [throw_defer]

#  if _CCCL_HAS_EXCEPTIONS()
  e = throw_defer->*[] {
    throw ::std::runtime_error("boom");
  };
  EXPECT(static_cast<bool>(e));
  try
  {
    ::std::rethrow_exception(e);
  }
  catch (const ::std::runtime_error& ex)
  {
    EXPECT(::std::string_view(ex.what()) == "boom");
    return;
  }
  EXPECT(false, "rethrow should have transferred control");
#  endif // _CCCL_HAS_EXCEPTIONS()
};

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
