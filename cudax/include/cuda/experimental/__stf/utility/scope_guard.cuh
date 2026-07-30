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
 * @brief Implements the SCOPE mechanism
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

#include <cuda/experimental/__stf/utility/unittest.cuh>
#include <cuda/experimental/__utility/scope_exit.cuh>

namespace cuda::experimental::stf
{
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
  auto CUDASTF_UNIQUE_NAME(scope_guard) = ::cuda::experimental::stf::detail::scope_guard_handler::kind{}->*[&]()
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

#  ifndef NDEBUG
template <class F>
void invoke_nothrow(F& f, ::cuda::std::source_location loc)
{
  // Forward the SCOPE call-site location into throw_proof.
  ::cuda::experimental::with_location<::cuda::experimental::throw_proof_t>{::cuda::experimental::throw_proof, loc}->*
    [&] {
      f();
    };
}
#  else // ^^^ !NDEBUG ^^^ / vvv NDEBUG vvv
template <class F>
void invoke_nothrow(F& f, ::cuda::std::source_location)
{
  f();
}
#  endif // NDEBUG

template <typename F>
auto operator->*(::cuda::experimental::with_location<exit> where, F&& f)
{
  static_assert(::std::is_void_v<decltype(::std::forward<F>(f)())>, "SCOPE requires a void-returning callable");

  struct result
  {
    F f;
    ::cuda::std::source_location loc;
    bool active = true;

    result(F&& f, ::cuda::std::source_location loc)
        : f(::std::forward<F>(f))
        , loc(loc)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , loc(rhs.loc)
        , active(rhs.active)
    {
      rhs.active = false;
    }

    ~result() noexcept
    {
      if (active)
      {
        invoke_nothrow(f, loc);
      }
    }
  };

  return result{::std::forward<F>(f), where.loc};
}

template <typename F>
auto operator->*(::cuda::experimental::with_location<fail> where, F&& f)
{
  static_assert(::std::is_void_v<decltype(::std::forward<F>(f)())>, "SCOPE requires a void-returning callable");

  struct result
  {
    F f;
    ::cuda::std::source_location loc;
    int exceptions;
    bool active = true;

    result(F&& f, ::cuda::std::source_location loc, int exceptions)
        : f(::std::forward<F>(f))
        , loc(loc)
        , exceptions(exceptions)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , loc(rhs.loc)
        , exceptions(rhs.exceptions)
        , active(rhs.active)
    {
      rhs.active = false;
    }

    ~result() noexcept
    {
      if (active && ::std::uncaught_exceptions() == exceptions)
      {
        invoke_nothrow(f, loc);
      }
    }
  };

  // Run only if an exception is in flight: uncaught count is one above creation-time count.
  return result{::std::forward<F>(f), where.loc, ::std::uncaught_exceptions() + 1};
}

template <typename F>
auto operator->*(::cuda::experimental::with_location<success> where, F&& f)
{
  static_assert(::std::is_void_v<decltype(::std::forward<F>(f)())>, "SCOPE requires a void-returning callable");

  struct result
  {
    F f;
    int exceptions;
    bool active = true;

    result(F&& f, int exceptions)
        : f(::std::forward<F>(f))
        , exceptions(exceptions)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , exceptions(rhs.exceptions)
        , active(rhs.active)
    {
      rhs.active = false;
    }

    // May throw — unlike exit/fail.
    ~result() noexcept(false)
    {
      if (active && ::std::uncaught_exceptions() == exceptions)
      {
        f();
      }
    }
  };

  (void) where; // location unused; success may throw so throw_proof does not apply
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
