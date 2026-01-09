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

namespace cuda::experimental::stf
{
/**
 * @brief Automatically runs code when a scope is exited (`SCOPE(exit)`), exited by means of an exception
 * (`SCOPE(fail)`), or exited normally (`SCOPE(success)`).
 *
 * The code controlled by `SCOPE(exit)` and `SCOPE(fail)` must not throw, otherwise the application will be terminated.
 * The code controlled by `SCOPE(exit)` may throw.
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
#define SCOPE(kind)                                                                      \
  auto CUDASTF_UNIQUE_NAME(scope_guard) =                                                \
    ::std::integral_constant<::cuda::experimental::stf::scope_guard_condition,           \
                             (::cuda::experimental::stf::scope_guard_condition::kind)>() \
      ->*[&]()
///@}

enum class scope_guard_condition
{
  exit,
  fail,
  success
};

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
template <scope_guard_condition cond, typename F>
auto operator->*(::std::integral_constant<scope_guard_condition, cond>, F&& f)
{
  struct result
  {
    result(F&& f, int threshold)
        : f(::std::forward<F>(f))
        , threshold(threshold)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , threshold(rhs.threshold)
    {
      // Disable call to lambda in rhs's destructor in all cases, we don't want double calls
      rhs.threshold = -2;
    }

    // Destructor (i.e. user's lambda) may throw exceptions if and only if we're in `SCOPE(success)`.
    ~result() noexcept(cond != scope_guard_condition::success)
    {
      // By convention, call always if threshold is -1, never if threshold < -1
      if (threshold == -1 || ::std::uncaught_exceptions() == threshold)
      {
        f();
      }
    }

  private:
    F f;
    int threshold;
  };

  // Threshold is -1 for SCOPE(exit), the same as current exceptions count for SCOPE(success), and 1 above the current
  // exception count for SCOPE(fail).
  return result(
    ::std::forward<F>(f),
    cond == scope_guard_condition::exit ? -1 : ::std::uncaught_exceptions() + (cond == scope_guard_condition::fail));
}
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
  assert(!done);

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
