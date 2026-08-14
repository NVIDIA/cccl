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
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__stf/utility/source_location.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#ifdef UNITTESTED_FILE
#  include <sstream>
#endif // UNITTESTED_FILE

namespace cuda::experimental::stf
{
/**
 * @brief A suppressing handler that reports an exception and carries on.
 *
 * `on_throw(notify) << callable` reports on `stderr`; `on_throw(notify, stream) << callable`
 * reports on any `std::ostream`. Reporting somewhere else entirely is a matter of writing
 * another handler, which this one doubles as the model for. An object rather than a function
 * because it is an overload set, which must travel as one value.
 *
 * The report carries the location and the exception's message, or "nonstandard exception" for
 * an exception that does not derive from `std::exception` (which reaches a handler as
 * `nullptr`). Both overloads return `std::ignore`, marking a suppressing handler.
 */
struct notify_t
{
  //! @brief Reports on `stderr` and flushes it.
  decltype(::std::ignore)
  operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc) const noexcept
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

  //! @brief Reports on `__os` and flushes it. The report is best-effort: a stream configured
  //! to throw does not get to end the program from inside a handler.
  decltype(::std::ignore) operator()(
    ::std::ostream& __os, const ::std::exception* __exception, const ::cuda::std::source_location __loc) const noexcept
  {
    _CCCL_TRY
    {
      __os << __loc.file_name() << '(' << __loc.line() << ") on_throw violation in " << __loc.function_name() << ": "
           << (__exception != nullptr ? __exception->what() : "nonstandard exception") << '\n';
      __os.flush();
    }
    _CCCL_CATCH_ALL {}
    return ::std::ignore;
  }
};
inline constexpr notify_t notify{};

/**
 * @brief The bottom type: a type with no values, convertible to every type.
 *
 * A callable that declares `nothing` as its return type promises in the type system that it
 * never returns normally: keeping the promise any other way would require materializing a value
 * of a type that has none. `[[noreturn]]` makes the same promise to the optimizer, but not
 * reliably to overload resolution; a `nothing` result states it as a fact of the type, visible
 * to metaprogramming and impossible to fake.
 *
 * The conversion operator lets a `nothing` expression appear wherever a value of any type is
 * expected, references included: a never-returning call may be `return`ed from a function of
 * any result type, or supply one arm of a ternary whose other arm produces the legitimate
 * value, as in `ready ? front() : abort()`. The operator can never run -- running it would
 * require an object that cannot exist -- so its body exists to satisfy the compiler, not to
 * execute.
 */
struct nothing final
{
  nothing()                          = delete;
  nothing(const nothing&)            = delete;
  nothing& operator=(const nothing&) = delete;

  // Two operators, because deduction for conversion functions strips the reference off the
  // target before matching: the rvalue one serves values and rvalue references, the lvalue one
  // serves lvalue references. A value target sees both and prefers the rvalue binding, so the
  // pair is not ambiguous. The bodies are unreachable rather than aborting: every `nothing`
  // prvalue is the result of a call that never returns, so control provably cannot arrive here
  // short of undefined behavior already committed elsewhere.
  template <class _Tp>
  [[noreturn]] _CCCL_HOST_DEVICE operator _Tp&&() const noexcept
  {
    _CCCL_UNREACHABLE();
  }
  template <class _Tp>
  [[noreturn]] _CCCL_HOST_DEVICE operator _Tp&() const noexcept
  {
    _CCCL_UNREACHABLE();
  }
};

/**
 * @brief `std::abort` wrapped so that never returning is part of its type: the reaction to
 * pass as in `on_throw(abort) << callable`.
 *
 * Inside this namespace, plain `abort` finds this function before the C library's. Code that
 * sees both through using-directives gets an ambiguity error rather than a silent pick, and
 * disambiguates with a using-declaration: `using cuda::experimental::stf::abort;`.
 */
[[noreturn]] inline nothing abort() noexcept
{
  ::std::abort(); // noreturn, so no value of the value-less result type is owed
}

//! @brief `std::terminate` wrapped like `abort`, its type proving it never returns.
[[noreturn]] inline nothing terminate() noexcept
{
  ::std::terminate();
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// Whether invoking _Fn with _Args... provably never returns, its declared result `nothing`
// having no values. The primary template reads as `false` for a non-invocable pairing; the
// partial specialization's void_t keeps `invoke_result_t` from hard-erroring in that case.
template <class _AlwaysVoid, class _Fn, class... _Args>
inline constexpr bool __never_returns_impl = false;

template <class _Fn, class... _Args>
inline constexpr bool
  __never_returns_impl<::cuda::std::void_t<::cuda::std::invoke_result_t<_Fn, _Args...>>, _Fn, _Args...> =
    ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<::cuda::std::invoke_result_t<_Fn, _Args...>>, nothing>;

template <class _Fn, class... _Args>
inline constexpr bool __never_returns = __never_returns_impl<void, _Fn, _Args...>;

// Binds a stream to a reaction that takes one as its first argument, the result being an
// ordinary handler. Produced by the stream-taking on_throw overload.
template <class _Reaction>
struct __stream_bound
{
  _Reaction __reaction_;
  ::std::ostream& __os_;

  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc) noexcept
  {
    static_assert(
      ::cuda::std::
        is_nothrow_invocable_v<_Reaction&, ::std::ostream&, const ::std::exception*, ::cuda::std::source_location>,
      "an on_throw reaction given a stream must be noexcept-invocable with "
      "(std::ostream&, const std::exception*, source_location)");
    return __reaction_(__os_, __exception, __loc);
  }
};

// One policy for all four reactions: recognizing them in one place keeps the order of the
// questions explicit, which matters because `nothing` converts to every type, so any
// convertibility question must be asked after the never-returns question. `operator<<` lives
// here too, since argument-dependent lookup only searches the innermost namespace enclosing
// the policy type.
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
    // The reactions whose call provably never returns, each in its own calling shape.
    constexpr bool __handler_dies = __never_returns<_Reaction&, const ::std::exception*, ::cuda::std::source_location>;
    constexpr bool __action_dies  = !__handles && __never_returns<_Reaction&>;
    // Whether the exception is answered rather than fatal, which `::std::ignore` decides by being
    // what it is and a handler decides by returning it. `nothing` converts to every type,
    // `::std::ignore`'s included, so a dying handler is asked by name first.
    constexpr bool __resumes =
      __handles
        ? !__handler_dies
            && ::cuda::std::
              is_invocable_r_v<decltype(::std::ignore), _Reaction&, const ::std::exception*, ::cuda::std::source_location>
        : __drops;

    if constexpr (__handles)
    {
      static_assert(
        ::cuda::std::is_nothrow_invocable_v<_Reaction&, const ::std::exception*, ::cuda::std::source_location>,
        "an on_throw handler runs while an exception is in flight and must be noexcept");
      // The return type is the handler's whole answer; in particular `void` is rejected, saying
      // nothing about what comes next.
      static_assert(__handler_dies || __resumes,
                    "an on_throw handler must return nothing, to end the program, or ::std::ignore, to go on");
      __reaction_(__exception, __loc_);
      if constexpr (__handler_dies)
      {
        // The call above cannot return: its declared result type has no values.
        _CCCL_UNREACHABLE();
      }
    }
    else if constexpr (__action_dies)
    {
      // A terminating action, `abort` and `terminate` being the ones provided. No backstop
      // follows the call: its result type already proves it cannot return, which is also what
      // lets it stand in for a callable returning a reference.
      notify(__exception, __loc_);
      __reaction_();
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
                    "an on_throw reaction is a handler, a never-returning callable (one returning "
                    "nothing, like abort and terminate), ::std::ignore, or a value convertible to "
                    "the result of the callable");
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
 *   is its whole answer to what happens next: `nothing` proves the handler never returns, and
 *   `decltype(::std::ignore)` resumes execution with a default-constructed result. Any other
 *   return type, `void` included, is rejected for saying nothing about what comes next.
 *   `notify` is such a handler.
 * - A terminating action: a callable invocable with no arguments whose declared result is
 *   `nothing`, of which `abort` and `terminate` above are the ones provided. The exception is
 *   reported through `notify`, then the action runs, its result type proof that it never
 *   returns. `std::abort` itself does not qualify -- a `void (*)()` says nothing about
 *   returning, and any captureless lambda converts to one -- so pass `abort`, or wrap your own
 *   ending in a `nothing`-returning callable.
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
 * A callable returning a reference therefore goes with a never-returning reaction, which never
 * has to produce a result, or with a replacement passed as an lvalue of the same type, which the
 * policy refers to rather than copies and which the caller keeps alive:
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

/**
 * @brief Creates a policy like `on_throw(reaction)` with a stream bound as the reaction's first
 * argument: `on_throw(notify, std::cerr) << callable` reports to `cerr`.
 *
 * The reaction must be a `noexcept` callable taking `(std::ostream&, const std::exception*,
 * cuda::std::source_location)`. Its return type keeps the usual handler meaning: `nothing` to
 * end the program, `std::ignore` to go on. `notify` has both this shape and the plain one.
 *
 * @param[in] __reaction The reaction, invoked with `__os` prepended to a handler's arguments.
 * @param[in] __os The stream to report to, which the caller keeps alive.
 * @param[in] __loc The location passed to the reaction; defaults to the call site.
 * @return A policy object consumed by `operator<<`.
 */
template <class _Reaction>
auto on_throw(_Reaction&& __reaction,
              ::std::ostream& __os,
              const ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
{
  return on_throw(detail::__stream_bound<_Reaction>{::cuda::std::forward<_Reaction>(__reaction), __os}, __loc);
}

#ifdef UNITTESTED_FILE
UNITTEST("nothing")
{
  using namespace cuda::experimental::stf;
  // No values: not constructible in any way.
  static_assert(!::std::is_default_constructible_v<nothing>);
  static_assert(!::std::is_copy_constructible_v<nothing>);
  static_assert(!::std::is_move_constructible_v<nothing>);
  // One-way conversions: `nothing` converts to every type, no type converts to `nothing`.
  static_assert(::std::is_convertible_v<nothing, int>);
  static_assert(::std::is_convertible_v<nothing, int&>);
  static_assert(::std::is_convertible_v<nothing, void (*)()>);
  static_assert(!::std::is_convertible_v<int, nothing>);
  // A never-returning call may be returned from a function of any result type, references
  // included; the conversion typechecks and never runs.
  [[maybe_unused]] const auto propagates = []() -> int& {
    return cuda::experimental::stf::abort();
  };
  // A `nothing` expression also supplies one arm of a ternary, the other arm setting the type.
  const auto pick = [](bool ok) -> int {
    return ok ? 42 : cuda::experimental::stf::abort();
  };
  EXPECT(pick(true) == 42);
};

UNITTEST("on_throw")
{
  using namespace cuda::experimental::stf;
  //! [on_throw]
  // The C library also declares ::abort, so under a using-directive the typed one is picked
  // by name; qualifying every use works as well.
  using cuda::experimental::stf::abort;
  int value = 0;
  on_throw(abort) << [&] {
    value = 42; // would abort the application if this code threw
  };
  on_throw(terminate) << [] {};
  on_throw(notify) << [] {}; // would report the exception on stderr and carry on
  const int answer = on_throw(-1) << [] {
    return 42; // would yield -1 instead if this code threw
  };
  EXPECT(value == 42);
  EXPECT(answer == 42);
  //! [on_throw]

  // A terminating handler declares `nothing` and dies on its own terms, no backstop needed;
  // it stays out of the way as long as nothing throws.
  const auto die = [](const ::std::exception*, ::cuda::std::source_location) noexcept -> nothing {
    ::std::abort();
  };
  const int untouched = on_throw(die) << [] {
    return 7;
  };
  EXPECT(untouched == 7);

  // Any callable whose declared result is `nothing` works as a terminating action; the type,
  // not an attribute, is what proves it never returns.
  const auto bail = []() noexcept -> nothing {
    ::std::abort();
  };
  const int spared = on_throw(bail) << [] {
    return 9;
  };
  EXPECT(spared == 9);

  // A never-returning reaction is also the one that goes with a reference result, since it
  // never has to produce one. The referent is static because nvcc reads a return of a
  // by-reference capture as a return of a local.
  static int target = 5;
  int& alias        = on_throw(abort) << []() -> int& {
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

  // A stream binds as the handler's first argument; notify itself has the shape.
  ::std::ostringstream stream_log;
  const int streamed = on_throw(notify, stream_log, site) << []() -> int {
    throw ::std::runtime_error("streamed");
  };
  EXPECT(streamed == 0);
  {
    char streamed_expected[1024]{};
    ::snprintf(streamed_expected,
               sizeof(streamed_expected),
               "%s(%u) on_throw violation in %s: streamed\n",
               site.file_name(),
               site.line(),
               site.function_name());
    EXPECT(stream_log.str() == streamed_expected);
  }

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
 * defined) those lambdas are invoked via `on_throw(abort)`; in release
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
  on_throw(abort, loc) << f;
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
