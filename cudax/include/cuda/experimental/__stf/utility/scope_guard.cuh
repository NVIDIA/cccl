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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

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
 * @brief A suppressing handler that reports an exception on `stderr` and flushes it.
 *
 * Use it as in `on_throw(notify) << callable` to report an exception and carry on.
 * Reporting to a destination other than `stderr` is a matter of writing another
 * handler, which this function doubles as the model for.
 *
 * @param[in] __exception The caught exception, or `nullptr` if it does not derive from
 *            `std::exception`, in which case the report carries no message.
 * @param[in] __loc The location to report.
 * @return `std::ignore`, which marks this handler as suppressing.
 */
inline decltype(::std::ignore)
notify(const ::std::exception* __exception, const ::cuda::std::source_location __loc) noexcept
{
  if (__exception != nullptr)
  {
    ::fprintf(stderr,
              "%s(%u) on_throw violation in %s: %s\n",
              __loc.file_name(),
              __loc.line(),
              __loc.function_name(),
              __exception->what());
  }
  else
  {
    ::fprintf(stderr,
              "%s(%u) on_throw violation in %s: nonstandard exception\n",
              __loc.file_name(),
              __loc.line(),
              __loc.function_name());
  }
  ::fflush(stderr);
  return ::std::ignore;
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// One policy for all four reactions: the alternative is constraining the `on_throw` overloads
// against one another, which is fragile, because clang keeps `[[noreturn]]` in the type of
// `std::abort` and thereby makes a `void (*)() noexcept` parameter an inexact match that a
// catch-all template would win. `operator<<` lives here too, since argument-dependent lookup
// only searches the innermost namespace enclosing the policy type.
template <class _Reaction>
struct __on_throw_policy
{
  _Reaction __reaction_;
  const ::cuda::std::source_location __loc_;

  // Runs with the exception in flight and produces what `operator<<` returns in its stead. The
  // reactions that say nothing never read the exception, which gcc 9 flags without the attribute.
  template <class _Result>
  _Result __handle([[maybe_unused]] const ::std::exception* __exception) noexcept
  {
    // A handler is anything callable with the caught exception and a location; the other three
    // reactions are recognized in the branches below.
    constexpr bool __handles =
      ::cuda::std::is_invocable_v<_Reaction&, const ::std::exception*, ::cuda::std::source_location>;
    constexpr bool __drops =
      ::cuda::std::is_same_v<const ::cuda::std::remove_reference_t<_Reaction>, const decltype(::std::ignore)>;
    // Whether the exception is answered rather than fatal, which `::std::ignore` decides by being
    // what it is and a handler decides by returning it. Asking the same of `void` would be
    // useless, every result type being convertible to `void`, but nothing except `::std::ignore`
    // itself converts to its type.
    constexpr bool __resumes =
      __handles
        ? ::cuda::std::
            is_invocable_r_v<decltype(::std::ignore), _Reaction&, const ::std::exception*, ::cuda::std::source_location>
        : __drops;

    if constexpr (__handles)
    {
      static_assert(
        ::cuda::std::is_nothrow_invocable_v<_Reaction&, const ::std::exception*, ::cuda::std::source_location>,
        "an on_throw handler runs while an exception is in flight and must be noexcept");
      __reaction_(__exception, __loc_);
      if constexpr (!__resumes)
      {
        static_assert(
          ::cuda::std::is_void_v<
            ::cuda::std::invoke_result_t<_Reaction&, const ::std::exception*, ::cuda::std::source_location>>,
          "an on_throw handler must return void, to end the program, or ::std::ignore, to go on");
        ::std::abort(); // the handler was supposed to see to that itself
        _CCCL_UNREACHABLE();
      }
    }
    else if constexpr (::cuda::std::is_convertible_v<_Reaction, void (*)()>)
    {
      // A terminating action, `::std::abort` and `::std::terminate` being the ones worth naming.
      // A `noexcept` pointer is not asked for, even though the action runs during unwinding,
      // because Microsoft's `abort` is not declared `noexcept` and would miss this branch.
      notify(__exception, __loc_);
      __reaction_();
      ::std::abort(); // in case it returns after all
      _CCCL_UNREACHABLE();
    }
    else if constexpr (!__drops)
    {
      // A reference result may only name the reaction itself: a value the policy owns dies with
      // this call, and so does any temporary a conversion would materialize. Pointers convert
      // exactly when the reference binds directly, which is the question being asked.
      static_assert(
        !::cuda::std::is_reference_v<_Result>
          || (::cuda::std::is_lvalue_reference_v<_Reaction> && ::cuda::std::is_lvalue_reference_v<_Result>
              && ::cuda::std::is_convertible_v<::cuda::std::remove_reference_t<_Reaction>*,
                                               ::cuda::std::remove_reference_t<_Result>*>),
        "a reference result needs an on_throw reaction passed as an lvalue of the same "
        "type, anything else dying with the call");
      static_assert(::cuda::std::is_convertible_v<_Reaction, _Result>,
                    "an on_throw reaction is a handler, something convertible to void (*)(), "
                    "::std::ignore, or a value convertible to the result of the callable");
      return ::cuda::std::forward<_Reaction>(__reaction_);
    }

    // Left over are the two reactions that resume, both of which owe the caller a result.
    if constexpr (__resumes && !::cuda::std::is_void_v<_Result>)
    {
      static_assert(!::cuda::std::is_reference_v<_Result>,
                    "an on_throw reaction that resumes has nothing to refer to for a reference result");
      static_assert(::cuda::std::is_default_constructible_v<_Result>,
                    "an on_throw reaction that resumes requires a default-constructible result");
      return _Result{};
    }
  }
};

template <class _Reaction, class _Fn>
decltype(auto) operator<<(__on_throw_policy<_Reaction> __policy, _Fn&& __fn) noexcept
{
  using _Result = decltype(::cuda::std::forward<_Fn>(__fn)());
  // A `noexcept` callable puts the reaction out of reach: an exception raised inside it ends the
  // program where it stands, so the catch below could never run and the policy would be a promise
  // nobody keeps.
  static_assert(!noexcept(::cuda::std::forward<_Fn>(__fn)()),
                "on_throw has nothing to do for a noexcept callable, which terminates rather than "
                "throws; call such a callable directly");
  _CCCL_TRY
  {
    return ::cuda::std::forward<_Fn>(__fn)();
  }
  _CCCL_CATCH (const ::std::exception& __exception)
  {
    return __policy.template __handle<_Result>(&__exception);
  }
  _CCCL_CATCH_ALL
  {
    return __policy.template __handle<_Result>(nullptr);
  }
}
} // namespace detail
#endif // !_CCCL_DOXYGEN_INVOKED

/**
 * @brief Creates a policy saying how to react if a callable throws.
 *
 * Apply the policy with `on_throw(reaction) << callable`, which evaluates to the result of the
 * callable when nothing goes wrong. The reaction is one of four things, recognized by type:
 *
 * - A handler: any `noexcept` callable accepting a `const std::exception*` and a
 *   `cuda::std::source_location`, be it a function, a function pointer, or a function object,
 *   capturing or not. It is invoked with the caught exception, or with `nullptr` for an
 *   exception that does not derive from `std::exception`, along with `__loc`. Its return type
 *   says what happens next: returning `decltype(::std::ignore)` resumes execution with a
 *   default-constructed result, and returning `void` claims the handler ends the program, with
 *   `std::abort` running right after it in case it does not. `notify` is such a handler.
 * - A terminating action: anything convertible to `void (*)()`, `std::abort` and
 *   `std::terminate` being the obvious ones. The exception is reported through `notify`, the
 *   action runs, and `std::abort` follows in case it returns.
 * - `std::ignore`, which suppresses the exception silently and resumes execution with a
 *   default-constructed result.
 * - Anything else, taken as a replacement value for the result. It must be convertible to the
 *   callable's result type, and is moved into the result if the policy owns it.
 *
 * The two resuming reactions leave a `void` result alone and require a default-constructible
 * type of a non-void one, having nothing to refer to for a reference.
 *
 * The callable itself must not be `noexcept`: an exception raised inside one ends the program
 * where it stands, leaving the reaction unreachable, so such a pairing is rejected instead of
 * standing there looking like protection. Call such a callable directly.
 *
 * A callable returning a reference therefore goes with a terminating action, which never has to
 * produce a result, or with a replacement passed as an lvalue of the same type, which the policy
 * refers to rather than copies and which the caller keeps alive:
 *
 * @code
 * int fallback = 42;
 * int& x = on_throw(fallback) << [] { return returns_a_reference(); }; // x is fallback on a throw
 * @endcode
 *
 * @param[in] __reaction The reaction, which the policy owns if passed an rvalue and refers to if
 *            passed an lvalue.
 * @param[in] __loc The location passed to the handler; defaults to the call site.
 * @return A policy object consumed by `operator<<`.
 */
template <class _Reaction>
auto on_throw(_Reaction&& __reaction,
              const ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
{
  return detail::__on_throw_policy<_Reaction>{::cuda::std::forward<_Reaction>(__reaction), __loc};
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
  on_throw(notify) << [] {}; // would report the exception on stderr and carry on
  const int answer = on_throw(-1) << [] {
    return 42; // would yield -1 instead if this code threw
  };
  EXPECT(value == 42);
  EXPECT(answer == 42);
  //! [on_throw]

  // A terminating handler, recognized by its void return, stays out of the way as long as
  // nothing throws.
  const auto die = [](const ::std::exception*, ::cuda::std::source_location) noexcept {
    ::std::abort();
  };
  const int untouched = on_throw(die) << [] {
    return 7;
  };
  EXPECT(untouched == 7);

  // A terminating action need not be `noexcept`, which is how Microsoft declares `abort`.
  void (*const bail)() = [] {
    ::std::abort();
  };
  const int spared = on_throw(bail) << [] {
    return 9;
  };
  EXPECT(spared == 9);

  // A terminating action is also the one reaction that goes with a reference result, since it
  // never has to produce one. The referent is static because nvcc reads a return of a
  // by-reference capture as a return of a local.
  static int target = 5;
  int& alias        = on_throw(::std::abort) << []() -> int& {
    return target;
  };
  EXPECT(&alias == &target);

  // A replacement passed as an lvalue outlives the call, so it can stand in for a reference
  // result as well.
  int fallback = 42;
  int& picked  = on_throw(fallback) << []() -> int& {
    return target;
  };
  EXPECT(&picked == &target);

#  if _CCCL_HAS_EXCEPTIONS()
  int& supplanted = on_throw(fallback) << []() -> int& {
    throw ::std::runtime_error("no reference to give");
  };
  EXPECT(&supplanted == &fallback);

  const int ignored = on_throw(::std::ignore) << []() -> int {
    throw ::std::runtime_error("ignored");
  };
  EXPECT(ignored == 0);
  on_throw(::std::ignore) << [] {
    throw 42;
  };

  // Reporting somewhere other than stderr is what defining a handler is for; `notify` is the
  // same code writing to stderr, which a test cannot capture portably. A handler may capture,
  // so the destination needs no static storage.
  ::FILE* const log = ::tmpfile();
  EXPECT(log != nullptr);
  const auto to_log = [log](const ::std::exception* __exception, ::cuda::std::source_location __where) noexcept {
    ::fprintf(log,
              "%s(%u) in %s: %s\n",
              __where.file_name(),
              __where.line(),
              __where.function_name(),
              __exception != nullptr ? __exception->what() : "nonstandard exception");
    return ::std::ignore;
  };

  const auto site  = ::cuda::std::source_location::current();
  const int logged = on_throw(to_log, site) << []() -> int {
    throw ::std::runtime_error("boom");
  };
  EXPECT(logged == 0);

  // An exception that does not derive from std::exception reaches the handler as nullptr.
  on_throw(to_log, site) << [] {
    throw 42;
  };

  ::rewind(log);
  char message[1024]{};
  char expected[1024]{};
  EXPECT(::fgets(message, sizeof(message), log) != nullptr);
  ::snprintf(expected, sizeof(expected), "%s(%u) in %s: boom\n", site.file_name(), site.line(), site.function_name());
  EXPECT(::std::string_view{message} == expected);
  EXPECT(::fgets(message, sizeof(message), log) != nullptr);
  ::snprintf(expected,
             sizeof(expected),
             "%s(%u) in %s: nonstandard exception\n",
             site.file_name(),
             site.line(),
             site.function_name());
  EXPECT(::std::string_view{message} == expected);
  ::fclose(log);

  // A replacement value stands in for the result, converted to the callable's result type.
  const int replaced = on_throw(42) << []() -> int {
    throw ::std::runtime_error("replaced");
  };
  EXPECT(replaced == 42);
  const double widened = on_throw(42) << []() -> double {
    throw 42;
  };
  EXPECT(widened == 42.0);

  // The value is moved into the result, so a move-only replacement works.
  struct movable
  {
    int v;
    explicit movable(int value_)
        : v(value_)
    {}
    movable(const movable&) = delete;
    movable(movable&&)      = default;
  };
  const movable moved = on_throw(movable{7}) << []() -> movable {
    throw 42;
  };
  EXPECT(moved.v == 7);
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
  static_assert(::cuda::std::is_void_v<decltype(f())>, "SCOPE requires a void-returning callable");
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
        : f(::cuda::std::forward<F>(f))
        , loc(loc)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , loc(rhs.loc)
        , exceptions(::cuda::std::exchange(rhs.exceptions, -1))
    {}

    ~result() noexcept
    {
      if (exceptions != -1)
      {
        invoke_nothrow(f, loc);
      }
    }
  };

  return result{::cuda::std::forward<F>(f), where.loc};
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
        : f(::cuda::std::forward<F>(f))
        , loc(loc)
        , exceptions(exceptions)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , loc(rhs.loc)
        , exceptions(::cuda::std::exchange(rhs.exceptions, -1))
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
  return result{::cuda::std::forward<F>(f), where.loc, ::std::uncaught_exceptions() + 1};
}

template <typename F>
auto operator->*(success, F&& f)
{
  // success may throw, so it does not go through invoke_nothrow; keep the same void check.
  static_assert(::cuda::std::is_void_v<decltype(::cuda::std::forward<F>(f)())>,
                "SCOPE requires a void-returning callable");

  struct result
  {
    F f;
    // Expected uncaught count, or -1 when disarmed by move.
    int exceptions;

    result(F&& f, int exceptions)
        : f(::cuda::std::forward<F>(f))
        , exceptions(exceptions)
    {}
    result(result&) = delete;
    result(result&& rhs)
        : f(mv(rhs.f))
        , exceptions(::cuda::std::exchange(rhs.exceptions, -1))
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

  return result{::cuda::std::forward<F>(f), ::std::uncaught_exceptions()};
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
