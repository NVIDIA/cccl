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

#include <cuda/cccl_config>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if defined(CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/exception/exception_macros.h>
#include <cuda/std/functional/invoke.h>
#include <cuda/std/type_traits/conditional.h>
#include <cuda/std/type_traits/decay.h>
#include <cuda/std/type_traits/enable_if.h>
#include <cuda/std/type_traits/is_base_of.h>
#include <cuda/std/type_traits/is_convertible.h>
#include <cuda/std/type_traits/is_default_constructible.h>
#include <cuda/std/type_traits/is_reference.h>
#include <cuda/std/type_traits/is_same.h>
#include <cuda/std/type_traits/is_valid_expansion.h>
#include <cuda/std/type_traits/is_void.h>
#include <cuda/std/type_traits/remove_cvref.h>
#include <cuda/std/utility/declval.h>
#include <cuda/std/utility/forward.h>
#include <cuda/std/utility/move.h>

#include <cuda/experimental/stf/utility/source_location.cuh>
#include <cuda/experimental/stf/utility/unittest.cuh>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <utility>

#ifdef UNITTESTED_FILE
#  include <sstream>
#  include <string>
#endif // UNITTESTED_FILE

// nvcc 12.0 hits an internal compiler error ("error while padding end of structure!") when a
// [[no_unique_address]] member's type is one of this header's empty policies; newer toolkits
// are fine. WAR: the attribute is applied only where the compiler survives it.
#if CCCL_CUDA_COMPILER(NVCC, <, 12, 1)
#  define CCCL_STF_NO_UNIQUE_ADDRESS
#else // ^^^ nvcc < 12.1 ^^^ / vvv other compilers vvv
#  define CCCL_STF_NO_UNIQUE_ADDRESS CCCL_NO_UNIQUE_ADDRESS
#endif // nvcc < 12.1

namespace cuda::experimental::stf
{
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
  template <class Tp>
  [[noreturn]] CCCL_HOST_DEVICE operator Tp&&() const noexcept
  {
    CCCL_UNREACHABLE();
  }
  template <class Tp>
  [[noreturn]] CCCL_HOST_DEVICE operator Tp&() const noexcept
  {
    CCCL_UNREACHABLE();
  }
};

/**
 * @brief Policy vocabulary for @ref on_throw.
 *
 * Nested and deliberately non-inline: the names are short English words, and
 * `using namespace cuda::experimental::stf;` is routine in user code -- it must not
 * acquire them. This is the `std::literals` design with the inline decision inverted:
 * whoever opens all of std wants its literals, while whoever opens stf is exactly who
 * this namespace protects from the policy vocabulary.
 */
namespace exception_policies
{
/**
 * @brief A suppressing handler policy that reports an exception and resumes (`std::ignore`).
 *
 * `on_throw(notify) << callable` reports on `stderr`. A configured copy reports elsewhere:
 * `notify(file)` writes to a `FILE*`, `notify(stream)` to a `std::ostream`. An object rather
 * than a function because it is an overload set, which must travel as one value.
 *
 * The report carries the location and the exception's message, or "nonstandard exception" for
 * an exception that does not derive from `std::exception` (which reaches a handler as
 * `nullptr`). The exception hook returns `std::ignore`, marking a resuming policy.
 */
struct notify_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  // Destination: `os_` wins if set, otherwise `file_` (default `stderr`).
  ::FILE* file_       = stderr;
  ::std::ostream* os_ = nullptr;

  //! @brief Returns a copy configured to report on `file` instead of `stderr`.
  notify_t operator()(::FILE* file) const
  {
    notify_t copy;
    copy.file_ = file;
    return copy;
  }

  //! @brief Returns a copy configured to report on `os` instead of `stderr`.
  notify_t operator()(::std::ostream& os) const
  {
    notify_t copy;
    copy.os_ = &os;
    return copy;
  }

  //! @brief Reports the exception and resumes. Writes to the configured `std::ostream` if any,
  //! else `file_` (default `stderr`). The ostream write is best-effort: a stream configured
  //! to throw does not get to end the program from inside a handler.
  template <class Fn>
  decltype(::std::ignore)
  operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn&) const noexcept
  {
    if (os_)
    {
      CCCL_TRY
      {
        *os_ << loc.file_name() << '(' << loc.line() << ") on_throw violation in " << loc.function_name()
               << ": " << (exception ? exception->what() : "nonstandard exception") << '\n';
        os_->flush();
      }
      CCCL_CATCH_ALL {}
    }
    else
    {
      ::fprintf(file_,
                "%s(%u) on_throw violation in %s: %s\n",
                loc.file_name(),
                loc.line(),
                loc.function_name(),
                exception ? exception->what() : "nonstandard exception");
      ::fflush(file_);
    }
    return ::std::ignore;
  }
};
// const rather than constexpr: the default destination `stderr` is not a constant expression.
inline const notify_t notify{};

/**
 * @brief Reporting ending: report through `notify`, then `std::abort`. Usable as
 * `on_throw(abort) << callable`, in ternaries (`ready ? front() : abort()`), and as a bare
 * call `abort()`.
 *
 * Inside `exception_policies`, plain `abort` finds this object before the C library's function.
 * Code that sees both through using-directives gets an ambiguity error rather than a silent
 * pick, and disambiguates with a using-declaration:
 * `using cuda::experimental::stf::exception_policies::abort;`. A block-scope using-declaration
 * still hides `::abort`.
 *
 * `notify & abort` reports twice (documented). `abort | p` is a dead-| error (hook is
 * noexcept); `abort & p` is a dead-& error (answers `nothing`).
 */
struct abort_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  //! @brief The bare call: usable in ternaries -- `ready ? front() : abort()`.
  [[noreturn]] nothing operator()() const noexcept
  {
    ::std::abort();
  }

  //! @brief The exception hook: report, then die.
  template <class Fn>
  [[noreturn]] nothing
  operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn) const noexcept
  {
    notify(exception, loc, fn);
    ::std::abort();
  }
};
inline constexpr abort_t abort{};

//! @brief Like @ref abort_t "abort", but ends via `std::terminate`.
struct terminate_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  [[noreturn]] nothing operator()() const noexcept
  {
    ::std::terminate();
  }

  template <class Fn>
  [[noreturn]] nothing
  operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn) const noexcept
  {
    notify(exception, loc, fn);
    ::std::terminate();
  }
};
inline constexpr terminate_t terminate{};

/**
 * @brief The identity element of `&`: a policy with no capabilities at all.
 *
 * `noop & p` and `p & noop` both behave as `p`. Its use is to head a chain so that every
 * binary application contains a policy this header defines, as in `noop & effect1 & effect2`,
 * since a chain of plain lambdas is not itself composable.
 */
struct noop_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond
};
inline constexpr noop_t noop{};

/**
 * @brief Capturing policy: `on_throw(defer) << callable` evaluates to a `std::exception_ptr`
 * that is empty when the callable returns normally and holds the thrown exception otherwise,
 * ready for storage and a later `std::rethrow_exception`. This is the policy for boundaries
 * that must not unwind but cannot decide either -- the exception's fate is somebody else's,
 * later.
 *
 * The callable must return `void`: the expression's value is the `exception_ptr`, leaving the
 * callable's result no channel. That requirement is expressed by providing only `on_success()`
 * and no `on_success(R&&)`: a non-void callable then finds no success hook that accepts its
 * result, which is the error surfaced.
 */
struct defer_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  //! @brief Captures the in-flight exception; the answer converts to the expression's type.
  template <class Fn>
  ::std::exception_ptr operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const noexcept
  {
    return ::std::current_exception();
  }

  //! @brief On success there is no exception, so the captured pointer is empty.
  ::std::exception_ptr on_success() const noexcept
  {
    return ::std::exception_ptr{};
  }
};
inline constexpr defer_t defer{};

/**
 * @brief The identity element of `|` and the decline primitive: re-throws the in-flight
 * exception from inside the catch. Its answer type is `nothing`, so it never has to produce a
 * value; being non-`noexcept` is how it declines, handing the exception to the next `|` arm or
 * letting it propagate.
 */
struct rethrow_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  template <class Fn>
  [[noreturn]] nothing operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const
  {
    throw;
  }
};
inline constexpr rethrow_t rethrow{};

/**
 * @brief Value-substitution policy; `subst(v)` is the documented spelling for what a bare
 * value passed to `on_throw` also means.
 *
 * The exception hook answers, in order: the result of invoking the stored value as a handler
 * `(const std::exception*, source_location, Fn&)` when that is well-formed (so
 * `subst([](const std::exception* e, auto, auto&){ ... })` reacts to the exception; a handler
 * stored in subst must not throw); else the result of invoking it as a nullary callable, a
 * lazy fallback computed only on the exception path (`subst([]{ return expensive(); })`); else
 * the stored value itself, forwarded out and owned if the policy owns it, referred to if it
 * holds an lvalue reference (so a replacement passed as an lvalue can stand in for a reference
 * result).
 */
template <class V>
struct subst_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  CCCL_STF_NO_UNIQUE_ADDRESS V v_;

  template <class Fn>
  decltype(auto) operator()([[maybe_unused]] const ::std::exception* exception,
                            [[maybe_unused]] const ::cuda::std::source_location loc,
                            [[maybe_unused]] Fn& fn) noexcept
  {
    if constexpr (::cuda::std::is_invocable_v<V&, const ::std::exception*, ::cuda::std::source_location, Fn&>)
    {
      return v_(exception, loc, fn);
    }
    else if constexpr (::cuda::std::is_invocable_v<V&>)
    {
      return v_();
    }
    else
    {
      return ::cuda::std::forward<V>(v_);
    }
  }
};

//! @brief Creates a value-substitution policy; see @ref subst_t.
template <class V>
auto subst(V&& v)
{
  return subst_t<V>{::cuda::std::forward<V>(v)};
}

/**
 * @brief The unit re-attempt: re-runs the callable once; if the re-run throws, declines with
 * that exception. Counts come from repetition: `retry * 3` re-attempts up to three times;
 * `retry * 3 | subst(fallback)` answers the spent failure. `retry` alone is `retry * 1`.
 *
 * Answers `fn()`'s value for a non-void callable, or `std::ignore` after a successful
 * re-run of a void callable. `&` discards non-final answers, so `retry & subst(0)` is legal
 * (and almost never what you want).
 */
struct retry_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  // decltype(auto), not auto: a reference-returning callable must re-run to the same object,
  // not to a copy (the ignore branch deduces a reference to the global, which never dangles).
  template <class Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, Fn& fn)
  {
    if constexpr (::cuda::std::is_void_v<decltype(fn())>)
    {
      fn(); // a throw here IS the decline
      return (::std::ignore); // resume: the void expression is complete
    }
    else
    {
      return fn();
    }
  }
};
inline constexpr retry_t retry{};

/**
 * @brief Typed expected-owner: `on_throw(expecting<E>) << f` yields
 * `cuda::std::expected<R, E>`. `E` may be ANY catchable type -- a std::exception derivative,
 * a user struct that never heard of std::exception, even `int` -- the hook re-observes the
 * active exception at type `E` rather than relying on the std::exception funnel. A polymorphic
 * `E` is matched by exact dynamic type (a derivative declines rather than slice); a
 * non-polymorphic `E` matches by ordinary catch-clause rules, exactly as a handwritten
 * `catch (const E&)` would. Anything that does not match declines by rethrowing. Pair with
 * `| subst` for a total policy. Owners cannot be `|` arms (an `unexpected` does not convert to
 * the callable's raw result); nest `on_throw` for that spelling.
 *
 * `expecting<std::exception_ptr>` is the total catch-everything form (also spelled
 * `as_expected`); the one corner it costs is that a literally-thrown `exception_ptr` object
 * cannot be type-matched.
 */
template <class E>
struct expecting_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  template <class R>
  ::cuda::std::expected<::cuda::std::decay_t<R>, E> on_success(R&& r) const
  {
    return ::cuda::std::expected<::cuda::std::decay_t<R>, E>{::cuda::std::in_place, ::cuda::std::forward<R>(r)};
  }

  template <class Void = void>
  ::cuda::std::expected<Void, E> on_success() const
  {
    return ::cuda::std::expected<Void, E>{};
  }

  template <class Fn>
  ::cuda::std::unexpected<E> operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const
  {
    CCCL_TRY
    {
      throw; // re-observe the active exception at type E
    }
    CCCL_CATCH (const E& caught)
    {
      if constexpr (::cuda::std::is_polymorphic_v<E>)
      {
        if (typeid(caught) != typeid(E))
        {
          throw; // a derivative: decline rather than slice
        }
      }
      return ::cuda::std::unexpected<E>{caught}; // by value, no allocation
    }
    CCCL_CATCH_FALLTHROUGH // no catch-all: an unmatched rethrow propagates, which IS the decline
    CCCL_UNREACHABLE();
  }
};

/**
 * @brief Total form: captures any exception as `std::exception_ptr` (the old `as_expected`).
 * Hook is noexcept; the dead-| theorem correctly bans it as a non-last `|` arm.
 */
template <>
struct expecting_t<::std::exception_ptr>
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  template <class R>
  ::cuda::std::expected<::cuda::std::decay_t<R>, ::std::exception_ptr> on_success(R&& r) const
  {
    return ::cuda::std::expected<::cuda::std::decay_t<R>, ::std::exception_ptr>{
      ::cuda::std::in_place, ::cuda::std::forward<R>(r)};
  }

  // Templates in name only: laziness keeps expected<void, exception_ptr> (and, through it,
  // bad_expected_access) from being instantiated in every including TU -- gcc 14/15 at -O3
  // report a spurious maybe-uninitialized inside the latter's inlined destructor.
  template <class Void = void>
  ::cuda::std::expected<Void, ::std::exception_ptr> on_success() const
  {
    return ::cuda::std::expected<Void, ::std::exception_ptr>{};
  }

  template <class Fn, class Eptr = ::std::exception_ptr>
  ::cuda::std::unexpected<Eptr>
  operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const noexcept
  {
    return ::cuda::std::unexpected<Eptr>{::std::current_exception()};
  }
};

template <class E>
inline constexpr expecting_t<E> expecting{};

//! @brief Baseline instance of `expecting<std::exception_ptr>`; see @ref expecting_t.
inline constexpr expecting_t<::std::exception_ptr> as_expected{};

/**
 * @brief Predicate guard for an exception-path sequence.
 *
 * There is no runtime "nop" answer on the exception path: a hook either accepts or declines.
 * A true predicate contributes a void effect answer so `&` continues; false declines by
 * throwing. As a `|` arm this means "not applicable, try the next arm"; inside `&`, false
 * declines the whole sequence. `catch_only` remains separate because its typed claims support
 * the starved-arm theorem, while arbitrary predicates do not.
 */
template <class Pred>
struct guard_t
{
  using exception_sink_tag = void;
  CCCL_STF_NO_UNIQUE_ADDRESS Pred pred_;

  template <class Fn>
  void operator()(const ::std::exception* exception, const ::cuda::std::source_location, Fn&)
  {
    if constexpr (::cuda::std::is_invocable_v<Pred&, const ::std::exception*>)
    {
      if (pred_(exception))
      {
        return;
      }
    }
    else
    {
      if (pred_())
      {
        return;
      }
    }
    throw; // decline: the guard does not apply
  }
};

//! @brief Creates a predicate guard; see @ref guard_t.
template <class Pred>
auto guard(Pred&& pred)
{
  return guard_t<Pred>{::cuda::std::forward<Pred>(pred)};
}

/**
 * @brief Translates the active exception by throwing the result of `fn(exception)`.
 *
 * A translation always declines, but with a different exception; a following `|` arm sees
 * the translated exception.
 */
template <class Fn>
struct translate_t
{
  using exception_sink_tag = void;
  CCCL_STF_NO_UNIQUE_ADDRESS Fn fn_;

  template <class Callable>
  [[noreturn]] nothing operator()(const ::std::exception* exception, const ::cuda::std::source_location, Callable&)
  {
    throw fn_(exception);
  }
};

//! @brief Creates an exception translator; see @ref translate_t.
template <class Fn>
auto translate(Fn&& fn)
{
  return translate_t<Fn>{::cuda::std::forward<Fn>(fn)};
}

/**
 * @brief Throws a stored exception with the active exception nested as its cause.
 */
template <class E>
struct nest_t
{
  using exception_sink_tag = void;
  CCCL_STF_NO_UNIQUE_ADDRESS E exception_;

  static_assert(::cuda::std::is_copy_constructible_v<E>, "nest(e) requires a copyable exception object");

  template <class Fn>
  [[noreturn]] nothing operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&)
  {
    CCCL_TRY
    {
      throw;
    }
    CCCL_CATCH_ALL
    {
      ::std::throw_with_nested(exception_);
    }
    CCCL_UNREACHABLE();
  }
};

//! @brief Stores an exception by value and nests the active exception beneath it.
template <class E>
auto nest(E&& exception)
{
  using Stored = ::cuda::std::decay_t<E>;
  return nest_t<Stored>{::cuda::std::forward<E>(exception)};
}

/**
 * @brief Effect policy that sleeps before the next element of an `&` sequence.
 *
 * `(delay(100ms) & retry) * 3` pauses before each re-attempt.
 */
template <class Duration>
struct delay_t
{
  using exception_sink_tag = void;
  CCCL_STF_NO_UNIQUE_ADDRESS Duration duration_;

  template <class Fn>
  void operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&)
  {
    ::std::this_thread::sleep_for(duration_);
  }
};

//! @brief Creates a sleeping effect policy; see @ref delay_t.
template <class Duration>
auto delay(Duration&& duration)
{
  return delay_t<Duration>{::cuda::std::forward<Duration>(duration)};
}

/**
 * @brief Re-attempts with decorrelated-jitter delays.
 *
 * Uses the AWS "Exponential Backoff and Jitter" decorrelated algorithm: plain exponential
 * backoff synchronizes clients into retry storms. Randomness is hook-local xorshift state;
 * there is no global state and no `<random>` dependency.
 */
struct backoff_t
{
  using exception_sink_tag = void;
  int n_;
  ::std::chrono::milliseconds initial_;

  template <class Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, Fn& fn)
  {
    if (n_ == 0)
    {
      throw;
    }

    const auto base = initial_.count();
    const auto cap  = base * 64;
    auto sleep      = base;
    auto state      = static_cast<unsigned long long>(::std::chrono::steady_clock::now().time_since_epoch().count());
    if (state == 0)
    {
      state = 1;
    }

    for (int attempt = 0; attempt < n_; ++attempt)
    {
      ::std::this_thread::sleep_for(::std::chrono::milliseconds{sleep});
      CCCL_TRY
      {
        if constexpr (::cuda::std::is_void_v<decltype(fn())>)
        {
          fn();
          return (::std::ignore);
        }
        else
        {
          return fn();
        }
      }
      CCCL_CATCH_ALL
      {
        if (attempt + 1 == n_)
        {
          throw;
        }
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const auto tripled = sleep * 3;
        const auto upper   = tripled < cap ? tripled : cap;
        const auto span    = upper - base + 1;
        sleep = base + static_cast<decltype(base)>(state % static_cast<unsigned long long>(span));
      }
    }
    CCCL_UNREACHABLE();
  }
};

//! @brief Creates a decorrelated-jitter retry policy.
inline backoff_t backoff(int n, ::std::chrono::milliseconds initial)
{
  CCCL_ASSERT(n >= 0, "backoff requires a non-negative retry count");
  return backoff_t{n, initial};
}

/**
 * @brief Serves the last successful value when a later call throws.
 *
 * The variable is held by reference and must be an lvalue. Success updates it and passes the
 * result through; failure substitutes the stored value.
 */
template <class T>
struct remember_t
{
  using exception_sink_tag = void;
  T& var_;

  template <class R>
  ::cuda::std::conditional_t<::cuda::std::is_lvalue_reference_v<R&&>, R&&, ::cuda::std::remove_cvref_t<R>>
  on_success(R&& result)
  {
    var_ = result;
    return ::cuda::std::forward<R>(result);
  }

  template <class Fn>
  T& operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const noexcept
  {
    return var_;
  }
};

//! @brief Creates a last-known-good policy from an lvalue.
template <class T>
auto remember(T& var)
{
  return remember_t<T>{var};
}

#ifndef CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// --- The policy protocol: two optional capabilities discovered by introspection ------------
//
// Each capability is one archetype alias + `IsValidExpansion`. The exception hook is always
// `(const std::exception*, source_location, Fn&)`; Fn-independent probes use a throwaway
// `void (&)()`. `hook_answer_t` IS the hook archetype -- one definition, both roles.

template <class P, class Fn>
using exception_hook_of = decltype(::cuda::std::declval<P&>()(
  ::cuda::std::declval<const ::std::exception*>(),
  ::cuda::std::declval<::cuda::std::source_location>(),
  ::cuda::std::declval<Fn&>()));

template <class P, class Fn = void (&)()>
inline constexpr bool has_exception_hook =
  ::cuda::std::IsValidExpansion<exception_hook_of, ::cuda::std::remove_reference_t<P>, Fn>::value;

template <class P, class Fn = void (&)()>
using hook_answer_t = exception_hook_of<::cuda::std::remove_reference_t<P>, Fn>;

// Capability 2a: the success hook `p.on_success(R&&)` for a given result type.
template <class P, class R>
using on_success_with_of = decltype(::cuda::std::declval<P&>().on_success(::cuda::std::declval<R>()));

template <class P, class R>
inline constexpr bool has_on_success_with =
  ::cuda::std::IsValidExpansion<on_success_with_of, ::cuda::std::remove_reference_t<P>, R>::value;

// Capability 2b: the nullary success hook `p.on_success()` (the void-result channel).
template <class P>
using on_success_void_of = decltype(::cuda::std::declval<P&>().on_success());

template <class P>
inline constexpr bool has_on_success_void =
  ::cuda::std::IsValidExpansion<on_success_void_of, ::cuda::std::remove_reference_t<P>>::value;

// A policy is anything exposing at least one capability (exception hook probed with a throwaway).
template <class P>
inline constexpr bool has_any_capability = has_exception_hook<P> || has_on_success_void<P>;

// Whether a type is one this header defines as an exception-sink policy, marked by the
// exception_sink_tag member. This is what &/| require of at least one operand, so they do
// not hijack unrelated types.
template <class P>
using exception_sink_tag_of = typename P::exception_sink_tag;

template <class P>
inline constexpr bool is_exception_sink_v =
  ::cuda::std::IsValidExpansion<exception_sink_tag_of, ::cuda::std::remove_cvref_t<P>>::value;

// Whether a reaction is `::std::ignore` (compared as the current code does).
template <class P>
inline constexpr bool is_ignore_v =
  ::cuda::std::is_same_v<const ::cuda::std::remove_cvref_t<P>,
                         const ::cuda::std::remove_reference_t<decltype(::std::ignore)>>;

// Whether a policy's answer type is `nothing` -- it never returns from the exception path. The
// two-step form keeps `hook_answer_t` from being named for a hookless policy: `&&` does not
// short-circuit template instantiation, so the answer is probed only in the `true` partial.
template <bool HasHook, class P, class Fn>
inline constexpr bool answers_nothing_impl = false;

template <class P, class Fn>
inline constexpr bool answers_nothing_impl<true, P, Fn> =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<hook_answer_t<P, Fn>>, nothing>;

template <class P, class Fn = void (&)()>
inline constexpr bool answers_nothing = answers_nothing_impl<has_exception_hook<P, Fn>, P, Fn>;

// Whether a policy's exception path is nothrow -- the same computation that
// `operator<<`'s conditional noexcept uses. Declining from `|` means throwing
// from that path, so a nothrow left side of `|` leaves the right unreachable.
template <class P, class Fn = void (&)()>
inline constexpr bool exception_path_nothrow_v =
  has_exception_hook<P, Fn>
  && ::cuda::std::is_nothrow_invocable_v<::cuda::std::remove_reference_t<P>&,
                                         const ::std::exception*,
                                         ::cuda::std::source_location,
                                         Fn&>;

// --- Adapters: normalize the historical reactions into policies ----------------------------

// `::std::ignore` as a policy: resume with a default-constructed result. Resume is
// polymorphic substitution of the default: on non-void callables this policy is equivalent to
// `subst([](auto*, auto, auto& fn) { return decltype(fn())(); })`; the marker exists because a
// void expression has no value to substitute, yet "resumed" and "merely an effect" must stay
// distinguishable answers.
struct ignore_policy
{
  using exception_sink_tag = void;

  template <class Fn>
  decltype(::std::ignore) operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const noexcept
  {
    return ::std::ignore;
  }
};

// Success-hook forwarding shared by the single-policy wrappers (`catch_only_t`,
// `as_policy`, `policy_pow`): both arities delegate to the wrapped policy, which owns
// the expression type.
template <class P>
struct forwards_success
{
  CCCL_STF_NO_UNIQUE_ADDRESS P p_;

  template <class R, ::cuda::std::enable_if_t<has_on_success_with<P, R>, int> = 0>
  decltype(auto) on_success(R&& r)
  {
    return p_.on_success(::cuda::std::forward<R>(r));
  }

  template <class Self = P, ::cuda::std::enable_if_t<has_on_success_void<Self>, int> = 0>
  decltype(auto) on_success()
  {
    return p_.on_success();
  }
};

// `A` claims `B` when a `catch (const A&)` clause would take a thrown `B`: same type, or
// publicly derived. is_same covers non-class types (is_base_of_v<int, int> is false).
template <class A, class B>
inline constexpr bool claims = ::cuda::std::is_same_v<A, B> || ::cuda::std::is_base_of_v<A, B>;

template <class B, class... As>
inline constexpr bool claimed_by_any = (claims<As, B> || ...);

// Intra-pack subsumption: reject when any listed type claims another (duplicates included).
// Message names the dead Derived entry.
template <class...>
inline constexpr bool catch_only_pack_ok = true;

template <class Head, class... Tail>
inline constexpr bool catch_only_pack_ok<Head, Tail...> =
  (!claims<Head, Tail> && ...) && (!claims<Tail, Head> && ...) && catch_only_pack_ok<Tail...>;

// `catch_only<E1, E2, ...>(p)`: run `p`'s exception path when the active exception matches ANY
// listed type by catch-clause rules (same or publicly derived), else decline by rethrowing.
// The listed types may be anything catchable, std::exception heritage or not; matching is by
// re-observation, since a pack cannot expand into sibling catch clauses. Native C++ has no
// multi-type catch clause; this adds expressivity the language lacks. Policy parameter leads
// so the exception-type pack trails.
template <class P, class... Es>
struct catch_only_t : forwards_success<P>
{
  using exception_sink_tag = void;

  // Does the active exception match any listed type? A recursive ladder of re-observations;
  // the binding must be named for the no-exceptions expansion of CCCL_CATCH.
  template <class E0, class... Rest>
  static bool matches_active()
  {
    CCCL_TRY
    {
      throw;
    }
    CCCL_CATCH (const E0& match)
    {
      static_cast<void>(match);
      return true;
    }
    CCCL_CATCH_ALL
    {
      if constexpr (sizeof...(Rest) > 0)
      {
        return matches_active<Rest...>();
      }
      else
      {
        return false;
      }
    }
  }

  template <class Fn, class Self = P, ::cuda::std::enable_if_t<has_exception_hook<Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
  {
    if (matches_active<Es...>())
    {
      // A matching non-std exception still reaches `P` as a null pointer, per the funnel.
      return this->p_(exception, loc, fn);
    }
    throw; // decline: no listed type claims the active exception
  }
};

// Tags a raw capability-bearing callable so normal forms are uniformly sink-typed.
// Forwards every capability it wraps; adds none. Storage follows the `subst_t<R>`
// convention: an lvalue reaction is held by reference, an rvalue moved in.
template <class P>
struct as_policy : forwards_success<P>
{
  using exception_sink_tag = void;

  template <class Fn, class Self = P, ::cuda::std::enable_if_t<has_exception_hook<Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
  {
    return this->p_(exception, loc, fn);
  }
};

// Normalize any reaction into a policy object, returned by value (first match wins). A
// value stored inside carries its own ref-ness (e.g. subst_t<int&> for a reference result).
// Every normal form is sink-tagged: raw capability-bearing callables wrap in `as_policy`.
template <class R>
auto normalize(R&& r)
{
  using P = ::cuda::std::remove_cvref_t<R>;
  if constexpr (is_exception_sink_v<P>)
  {
    // Already sink-tagged (named policies, composites, adapters, noop, ...).
    return static_cast<P>(::cuda::std::forward<R>(r));
  }
  else if constexpr (has_any_capability<P>)
  {
    // Raw callable with a capability: tag it so identity elimination and &/| stay ADL-findable.
    return as_policy<R>{::cuda::std::forward<R>(r)};
  }
  else if constexpr (is_ignore_v<P>)
  {
    return ignore_policy{};
  }
  else
  {
    return subst_t<R>{::cuda::std::forward<R>(r)};
  }
}

// The type `normalize` would produce for `R`. After `as_policy`, every normal form is
// sink-tagged, so identity elimination fires for every operand (ADDENDUM-1 §A.5 superseded).
template <class R>
using normalized_t = decltype(normalize(::cuda::std::declval<R>()));

template <class R>
inline constexpr bool normalizes_to_exception_sink_v = is_exception_sink_v<normalized_t<R>>;

// Success-hook forwarding shared by the `&` and `|` composites: rightmost-wins (the right
// element owns the expression type). The exception hook -- where the two composites differ --
// lives in the derived types.
template <class L, class R>
struct composite_hooks
{
  CCCL_STF_NO_UNIQUE_ADDRESS L l_;
  CCCL_STF_NO_UNIQUE_ADDRESS R r_;

  template <class Rr,
            ::cuda::std::enable_if_t<has_on_success_with<R, Rr> || has_on_success_with<L, Rr>, int> = 0>
  decltype(auto) on_success(Rr&& r)
  {
    if constexpr (has_on_success_with<R, Rr>)
    {
      return r_.on_success(::cuda::std::forward<Rr>(r));
    }
    else
    {
      return l_.on_success(::cuda::std::forward<Rr>(r));
    }
  }

  template <class LL                                                                               = L,
            class RR                                                                               = R,
            ::cuda::std::enable_if_t<has_on_success_void<RR> || has_on_success_void<LL>, int> = 0>
  decltype(auto) on_success()
  {
    if constexpr (has_on_success_void<R>)
    {
      return r_.on_success();
    }
    else
    {
      return l_.on_success();
    }
  }
};

// The sequencing composite `L & R`: on the exception path run `L` then `R`; `R` answers.
// `&` discards non-final answers -- a non-final `retry` re-runs and discards, legal and almost
// never what you want.
template <class L, class R>
struct policy_and : composite_hooks<L, R>
{
  using exception_sink_tag = void;

  static_assert(!answers_nothing<L>, "policies after a never-returning policy are unreachable");

  // Present iff either side has a hook. `L` fires (answer discarded), then `R` answers; with
  // no `R` hook the composite's answer is `void`, which `interpret_answer` rejects in final
  // position -- correct, since such a chain cannot answer on its own.
  template <class Fn,
            class LL                                                                             = L,
            class RR                                                                             = R,
            ::cuda::std::enable_if_t<has_exception_hook<LL> || has_exception_hook<RR>, int> = 0>
  decltype(auto) operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
  {
    if constexpr (has_exception_hook<L>)
    {
      static_cast<void>(this->l_(exception, loc, fn));
      if constexpr (has_exception_hook<R>)
      {
        return this->r_(exception, loc, fn);
      }
    }
    else
    {
      return this->r_(exception, loc, fn);
    }
  }
};

template <class L, class R>
policy_and(L, R) -> policy_and<L, R>;

// Forward declaration: `|` and `*` reuse this for arm answer interpretation (defined below).
template <class Expr, class P, class Fn>
Expr interpret_answer(
  P& policy, const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn);

// The left arm of `|` provably starves the right when: both are catch_only wrappers (or the
// right is a typed expecting), the left's guard list claims every type the right lists, and
// the left's inner policy never declines a claimed exception. Sound and incomplete, like
// every dead-code theorem here: nested composites and raw guards escape the pattern; an
// inner policy that can decline (nothrow test fails) keeps the right arm live.
template <class L, class R>
inline constexpr bool right_arm_starved = false;

template <class P1, class... As, class P2, class... Bs>
inline constexpr bool right_arm_starved<catch_only_t<P1, As...>, catch_only_t<P2, Bs...>> =
  exception_path_nothrow_v<P1> && (claimed_by_any<Bs, As...> && ...);

template <class P1, class... As, class E>
inline constexpr bool right_arm_starved<catch_only_t<P1, As...>, expecting_t<E>> =
  exception_path_nothrow_v<P1> && claimed_by_any<E, As...>;

// The alternation composite `L | R`: `L` claims first; if it declines by throwing, `R`
// handles the original (re-observed) exception. Each arm is called at the uniform 3-arg shape;
// acceptance is interpreted at `decltype(fn())`. A plain arm's answer must therefore be
// interpretable at the callable's result type -- `retry | subst(-1)` works, but
// `retry | as_expected` does not (an `unexpected` does not convert to the raw result);
// nest `on_throw(as_expected) << [&]{ return on_throw(retry * n) << f; }` for that spelling.
template <class L, class R>
struct policy_or : composite_hooks<L, R>
{
  using exception_sink_tag = void;

  static_assert(has_exception_hook<L> && has_exception_hook<R>,
                "both sides of | must answer the exception path (have an exception hook)");
  static_assert(!exception_path_nothrow_v<L>,
                "the left policy never declines; alternatives after it are unreachable");
  static_assert(!right_arm_starved<L, R>,
                "the left catch_only already claims every exception type the right arm lists; "
                "the right alternative is unreachable");

  template <class Fn,
            class LL                                                                             = L,
            class RR                                                                             = R,
            ::cuda::std::enable_if_t<has_exception_hook<LL> && has_exception_hook<RR>, int> = 0>
  decltype(auto) operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
  {
    using Raw = decltype(fn());

    const auto right = [&](const ::std::exception* cur) -> Raw {
      return interpret_answer<Raw>(this->r_, cur, loc, fn);
    };
    const auto reobserve_right = [&]() -> Raw {
      CCCL_TRY
      {
        throw;
      }
      CCCL_CATCH (const ::std::exception& e)
      {
        return right(&e);
      }
      CCCL_CATCH_ALL
      {
        return right(nullptr);
      }
    };

    // When the left arm owns the expression type (on_success), both arms answer in that owned
    // type: a match wraps the left's unexpected, a decline lifts the right arm's raw value
    // through on_success. This is what makes `expecting<E> | subst(v)` work.
    if constexpr (has_on_success_with<L, Raw>)
    {
      using Owned = decltype(::cuda::std::declval<L&>().on_success(::cuda::std::declval<Raw>()));
      CCCL_TRY
      {
        return Owned{this->l_(exception, loc, fn)};
      }
      CCCL_CATCH_ALL
      {
        return this->l_.on_success(reobserve_right());
      }
    }
    else if constexpr (::cuda::std::is_void_v<Raw> && has_on_success_void<L>)
    {
      using Owned = decltype(::cuda::std::declval<L&>().on_success());
      CCCL_TRY
      {
        return Owned{this->l_(exception, loc, fn)};
      }
      CCCL_CATCH_ALL
      {
        reobserve_right();
        return this->l_.on_success();
      }
    }
    else
    {
      // Neither arm owns: interpret each at the callable's result type. Void callables surface
      // ignore so this composite can still sit as a top-level policy.
      CCCL_TRY
      {
        if constexpr (::cuda::std::is_void_v<Raw>)
        {
          interpret_answer<Raw>(this->l_, exception, loc, fn);
          return (::std::ignore);
        }
        else
        {
          return interpret_answer<Raw>(this->l_, exception, loc, fn);
        }
      }
      CCCL_CATCH_ALL
      {
        if constexpr (::cuda::std::is_void_v<Raw>)
        {
          reobserve_right();
          return (::std::ignore);
        }
        else
        {
          return reobserve_right();
        }
      }
    }
  }
};

template <class L, class R>
policy_or(L, R) -> policy_or<L, R>;

// `p * n`: behaviorally the n-fold `|` of p with itself. One stored policy, invoked up to n
// times; the active exception is re-observed between iterations exactly as `policy_or` does
// between arms. `n == 0` declines immediately (the empty fold is rethrow). The stored policy's
// hook is invoked up to n times; with the inventory now stateless this needs no copying --
// user-defined policies should likewise tolerate re-invocation.
template <class P>
struct policy_pow : forwards_success<P>
{
  using exception_sink_tag = void;
  int n_;

  static_assert(has_exception_hook<P>,
                "the repeated policy must answer the exception path (have an exception hook)");
  static_assert(!exception_path_nothrow_v<P>,
                "the repeated policy never declines; repetitions after the first are unreachable");

  template <class Fn>
  decltype(auto) operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
  {
    using Expr = decltype(fn());
    if (n_ == 0)
    {
      throw; // empty fold: decline with the still-active exception
    }

    // Recurse inside the catch so the re-observed exception pointer stays alive for the
    // next arm (same lifetime rule as `policy_or`). The recursion is bounded: `left`
    // decreases every level and `left == 1` declines by rethrowing. gcc 14.3+/15's
    // -Winfinite-recursion is blind to exceptional exits and misreads instantiations whose
    // only normal returns are the recursive calls (e.g. a never-returning repeated policy).
    CCCL_DIAG_PUSH
    CCCL_DIAG_SUPPRESS_GCC("-Wpragmas") // gcc < 12 does not know the warning below; without this
                                         // line the unknown name itself trips -Werror=pragmas
    CCCL_DIAG_SUPPRESS_GCC("-Winfinite-recursion")
    const auto go = [&](auto& self, const ::std::exception* cur, int left) -> Expr {
      CCCL_TRY
      {
        return interpret_answer<Expr>(this->p_, cur, loc, fn);
      }
      CCCL_CATCH_ALL
      {
        if (left == 1)
        {
          throw;
        }
        CCCL_TRY
        {
          throw;
        }
        CCCL_CATCH (const ::std::exception& e)
        {
          return self(self, &e, left - 1);
        }
        CCCL_CATCH_ALL
        {
          return self(self, nullptr, left - 1);
        }
      }
    };
    if constexpr (::cuda::std::is_void_v<Expr>)
    {
      go(go, exception, n_);
      return (::std::ignore);
    }
    else
    {
      return go(go, exception, n_);
    }
    CCCL_DIAG_POP
  }
};

// Interpret the final element's answer as the expression's value, converting to `Expr`.
template <class Expr, class P, class Fn>
Expr interpret_answer(
  P& policy, const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
{
  using Answer = hook_answer_t<P, Fn>;
  static_assert(!::cuda::std::is_void_v<Answer>,
                "the final policy must answer the exception path: nothing to die, ::std::ignore "
                "to resume, or a value to substitute");

  if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<Answer>, nothing>)
  {
    // Never returns: no backstop beyond the unreachable marker.
    policy(exception, loc, fn);
    CCCL_UNREACHABLE();
  }
  else if constexpr (is_ignore_v<Answer>)
  {
    // Resume: default-construct the expression's value (nothing to do for void).
    static_cast<void>(policy(exception, loc, fn));
    if constexpr (!::cuda::std::is_void_v<Expr>)
    {
      static_assert(!::cuda::std::is_reference_v<Expr>,
                    "an on_throw reaction that resumes has nothing to refer to for a reference result");
      static_assert(::cuda::std::is_default_constructible_v<Expr>,
                    "an on_throw reaction that resumes requires a default-constructible result");
      return Expr{};
    }
  }
  else
  {
    // Substitute: convert the answer to the expression's type. A reference result may be served
    // only by an lvalue answer of a compatible type; anything else dies with the call.
    static_assert(
      !::cuda::std::is_reference_v<Expr>
        || (::cuda::std::is_lvalue_reference_v<Answer> && ::cuda::std::is_lvalue_reference_v<Expr>
            && ::cuda::std::is_convertible_v<::cuda::std::remove_reference_t<Answer>*,
                                             ::cuda::std::remove_reference_t<Expr>*>),
      "a reference result needs an on_throw reaction passed as an lvalue of the same "
      "type, anything else dying with the call");
    static_assert(::cuda::std::is_convertible_v<Answer, Expr>,
                  "an on_throw reaction is a policy, a never-returning callable (one returning "
                  "nothing, like abort and terminate), ::std::ignore, or a value convertible to "
                  "the result of the callable");
    return static_cast<Expr>(policy(exception, loc, fn));
  }
}

// Walk the chain on the exception path: with no answering hook anywhere, propagate; else
// interpret. The parameters go unread in the propagate instantiation, which gcc 9 flags
// without the attribute.
template <class Expr, class P, class Fn>
Expr on_exception(P& policy,
                     [[maybe_unused]] const ::std::exception* exception,
                     [[maybe_unused]] const ::cuda::std::source_location loc,
                     [[maybe_unused]] Fn& fn)
{
  if constexpr (!has_exception_hook<P, Fn>)
  {
    throw; // no element answered: let the exception propagate
  }
  else
  {
    return interpret_answer<Expr>(policy, exception, loc, fn);
  }
}

// The policy carrier. ADL finds `operator<<` here since the type lives in this namespace.
template <class Reaction>
struct on_throw_policy
{
  CCCL_STF_NO_UNIQUE_ADDRESS Reaction reaction_;
  const ::cuda::std::source_location loc_;
};

template <class R>
on_throw_policy(R, ::cuda::std::source_location) -> on_throw_policy<R>;

template <class Reaction, class Fn>
// A resuming chain reads neither exception nor location in some instantiations; gcc 9 flags the
// unread policy without the attribute.
decltype(auto) operator<<([[maybe_unused]] on_throw_policy<Reaction> policy,
                          Fn&& fn) noexcept(exception_path_nothrow_v<Reaction, Fn>)
{
  // Bind as a non-const lvalue: a hook may invoke it again later.
  Fn& f = fn;

  // A `noexcept` callable puts the policy out of reach: an exception raised inside it ends the
  // program where it stands, so the catch below could never run and the policy would be a
  // promise nobody keeps.
  static_assert(!noexcept(f()),
                "on_throw has nothing to do for a noexcept callable, which terminates rather than "
                "throws; call such a callable directly");

  using Result = decltype(f());
  using P      = Reaction;

  if constexpr (::cuda::std::is_void_v<Result>)
  {
    if constexpr (has_on_success_void<P>)
    {
      // A success hook owns the type (e.g. defer, as_expected): the expression is what it makes.
      using Expr = decltype(::cuda::std::declval<P&>().on_success());
      CCCL_TRY
      {
        f();
        return policy.reaction_.on_success();
      }
      CCCL_CATCH (const ::std::exception& exception)
      {
        return detail::on_exception<Expr>(policy.reaction_, &exception, policy.loc_, f);
      }
      CCCL_CATCH_ALL
      {
        return detail::on_exception<Expr>(policy.reaction_, nullptr, policy.loc_, f);
      }
    }
    else
    {
      // No success hook: the expression is void, the result passing through.
      CCCL_TRY
      {
        f();
      }
      CCCL_CATCH (const ::std::exception& exception)
      {
        return detail::on_exception<void>(policy.reaction_, &exception, policy.loc_, f);
      }
      CCCL_CATCH_ALL
      {
        return detail::on_exception<void>(policy.reaction_, nullptr, policy.loc_, f);
      }
    }
  }
  else if constexpr (has_on_success_with<P, Result>)
  {
    // A success hook transforms/owns the non-void result; the expression is its return type.
    using Expr = decltype(::cuda::std::declval<P&>().on_success(::cuda::std::declval<Result>()));
    CCCL_TRY
    {
      return policy.reaction_.on_success(f());
    }
    CCCL_CATCH (const ::std::exception& exception)
    {
      return detail::on_exception<Expr>(policy.reaction_, &exception, policy.loc_, f);
    }
    CCCL_CATCH_ALL
    {
      return detail::on_exception<Expr>(policy.reaction_, nullptr, policy.loc_, f);
    }
  }
  else
  {
    // No hook accepts the non-void result: pass it through, unless a void-only success hook
    // (e.g. defer over a non-void callable) means the result has no channel.
    static_assert(!has_on_success_void<P>, "the policy's on_success cannot accept the callable's result");
    CCCL_TRY
    {
      return f();
    }
    CCCL_CATCH (const ::std::exception& exception)
    {
      return detail::on_exception<Result>(policy.reaction_, &exception, policy.loc_, f);
    }
    CCCL_CATCH_ALL
    {
      return detail::on_exception<Result>(policy.reaction_, nullptr, policy.loc_, f);
    }
  }
}
} // namespace detail
#endif // !CCCL_DOXYGEN_INVOKED

/**
 * @brief Restricts a policy to exceptions matching any of `E1, E2, ...`: `catch_only<E...>(p)`
 * runs `p`'s exception path when the active exception matches any listed type by catch-clause
 * rules (same or publicly derived), and otherwise declines by rethrowing. The listed types may
 * be anything catchable -- std::exception derivatives, user structs, even `int`. Native C++
 * has no multi-type catch clause; this adds that expressivity. A matching exception that does
 * not derive from `std::exception` reaches `p`'s hook as a null pointer. A pack where one type
 * claims another (identical or base-of) is rejected -- the claimed entry would be dead.
 */
template <class... Es, class P>
auto catch_only(P&& p)
{
  static_assert(sizeof...(Es) > 0, "catch_only requires at least one exception type");
  static_assert(detail::catch_only_pack_ok<Es...>,
                "catch_only<..., Base, ..., Derived, ...>: the Derived entry is dead "
                "(Base already claims it)");
  auto np = detail::normalize(::cuda::std::forward<P>(p));
  return detail::catch_only_t<decltype(np), Es...>{::cuda::std::move(np)};
}

/**
 * @brief Sequences two policies: on the exception path `L` runs then `R`, and `R`'s answer
 * decides. `noop` is the two-sided identity. Constrained so at least one operand is a policy
 * this header defines, so it never hijacks unrelated `&` expressions; a chain of plain lambdas
 * is therefore not composable, but heading it with `noop` makes it so.
 */
template <class L,
          class R,
          ::cuda::std::enable_if_t<detail::is_exception_sink_v<L> || detail::is_exception_sink_v<R>, int> = 0>
auto operator&(L&& l, R&& r)
{
  return detail::policy_and{
    detail::normalize(::cuda::std::forward<L>(l)), detail::normalize(::cuda::std::forward<R>(r))};
}

//! @brief Left identity of `&`: `noop & p` is `normalize(p)` when that result is sink-tagged.
template <class R, ::cuda::std::enable_if_t<detail::normalizes_to_exception_sink_v<R>, int> = 0>
auto operator&(noop_t, R&& r)
{
  return detail::normalize(::cuda::std::forward<R>(r));
}

//! @brief Right identity of `&`. `noop` itself is excluded so `noop & noop` is not ambiguous.
template <class L,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<L>, noop_t>
                                     && detail::normalizes_to_exception_sink_v<L>,
                                   int> = 0>
auto operator&(L&& l, noop_t)
{
  return detail::normalize(::cuda::std::forward<L>(l));
}

/**
 * @brief Alternation: `L` gets first claim; if it declines by throwing, `R` handles the
 * original exception. `rethrow` is the two-sided identity. Same operand constraint as `&`.
 */
template <class L,
          class R,
          ::cuda::std::enable_if_t<detail::is_exception_sink_v<L> || detail::is_exception_sink_v<R>, int> = 0>
auto operator|(L&& l, R&& r)
{
  return detail::policy_or{
    detail::normalize(::cuda::std::forward<L>(l)), detail::normalize(::cuda::std::forward<R>(r))};
}

//! @brief Left identity of `|`: `rethrow | p` is `normalize(p)` when that result is sink-tagged.
template <class R, ::cuda::std::enable_if_t<detail::normalizes_to_exception_sink_v<R>, int> = 0>
auto operator|(rethrow_t, R&& r)
{
  return detail::normalize(::cuda::std::forward<R>(r));
}

//! @brief Right identity of `|`. `rethrow` itself is excluded so `rethrow | rethrow` is not ambiguous.
template <class L,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<L>, rethrow_t>
                                     && detail::normalizes_to_exception_sink_v<L>,
                                   int> = 0>
auto operator|(L&& l, rethrow_t)
{
  return detail::normalize(::cuda::std::forward<L>(l));
}

/**
 * @brief Repetition: `p * n` is the n-fold `|` of `p` with itself -- behaviorally
 * `p | p | ... | p` (n copies). Laws: `p * 0` ≡ `rethrow` (empty fold); `p * 1` ≡ `p`
 * (behaviorally); `p * (m + n)` ≡ `p * m | p * n`. `*` binds tighter than `&` and `|`, so
 * `(notify & retry) * 3` notifies before each re-attempt, while `notify & retry * 3`
 * notifies once then re-attempts three times.
 */
template <class P,
          ::cuda::std::enable_if_t<detail::is_exception_sink_v<::cuda::std::remove_cvref_t<P>>
                                     || detail::has_any_capability<::cuda::std::remove_cvref_t<P>>,
                                   int> = 0>
auto operator*(P&& p, int n)
{
  CCCL_ASSERT(n >= 0, "repetition requires a non-negative count");
  using Np = detail::normalized_t<P>;
  return detail::policy_pow<Np>{{detail::normalize(::cuda::std::forward<P>(p))}, n};
}

//! @brief Commuted form of `operator*`: `n * p` is `p * n`.
template <class P,
          ::cuda::std::enable_if_t<detail::is_exception_sink_v<::cuda::std::remove_cvref_t<P>>
                                     || detail::has_any_capability<::cuda::std::remove_cvref_t<P>>,
                                   int> = 0>
auto operator*(int n, P&& p)
{
  return ::cuda::std::forward<P>(p) * n;
}

/**
 * @brief Restricts a policy with a runtime predicate.
 *
 * Returns `guard(pred) & p`: as a `|` arm, false means the next alternative gets the
 * exception; inside a larger `&`, false declines the whole sequence.
 */
template <class Pred, class P>
auto when(Pred&& pred, P&& p)
{
  return ::cuda::experimental::stf::exception_policies::guard(::cuda::std::forward<Pred>(pred))
       & detail::normalize(::cuda::std::forward<P>(p));
}
} // namespace exception_policies

// Tripwire: the abort policy moved to exception_policies. An
// unqualified `abort` here would silently find ::abort (die with no
// report). Any such use must fail to compile instead.
template <class... Ts>
void abort(Ts&&...) = delete;

/**
 * @brief Creates a policy saying how to react if a callable throws.
 *
 * Apply the policy with `on_throw(policy) << callable`, which evaluates to the callable's
 * result when nothing goes wrong -- or to a type owned by a success hook, such as the
 * `std::exception_ptr` of `defer` or the `cuda::std::expected` of `as_expected`.
 *
 * A policy is an object exposing any of two optional capabilities, discovered by compile-time
 * introspection: the exception hook
 * `(const std::exception*, source_location, Fn&)` whose return value is its answer on the throw
 * path (the callable may be re-invoked by policies like `retry`; most policies ignore it), and
 * a success hook `on_success(...)` that observes or replaces the result. The named policies
 * include @ref notify_t "notify", @ref subst_t "subst", @ref defer_t "defer",
 * @ref rethrow_t "rethrow", @ref retry_t "retry", @ref expecting_t "expecting" /
 * @ref as_expected, @ref noop_t "noop", @ref catch_only, @ref guard_t "guard" / @ref when,
 * @ref translate_t "translate" / @ref nest, @ref delay_t "delay", @ref backoff, and
 * @ref remember_t "remember". Guards decline what they do not claim; translators decline with
 * a different exception; delay/backoff/retry re-run; remember serves the last success.
 * Policies compose with `&` (sequence; the last element answers; non-final answers are
 * discarded) and `|` (alternation; the left may decline by throwing), and with `*` (n-fold
 * `|`).
 *
 * For backward compatibility `on_throw` also accepts non-policy reactions: `std::ignore`
 * resumes with a default-constructed result; and anything else is taken as a substitution
 * value, exactly as `subst(value)` (including a user's nullary `nothing`-returning ending,
 * which dies silently -- pair with `notify &` to opt the report back in). A substitution
 * passed as an lvalue can serve a reference result, which the policy refers to rather than
 * copies:
 *
 * @code
 * int fallback = 42;
 * int& x = on_throw(fallback) << [] { return returns_a_reference(); }; // x is fallback on a throw
 * @endcode
 *
 * The callable itself must not be `noexcept`: an exception raised inside one ends the program
 * where it stands, leaving the policy unreachable, so such a pairing is rejected instead of
 * standing there looking like protection. Call such a callable directly.
 *
 * The location defaults to the call site; pass one explicitly to report a different site.
 *
 * Vocabulary visibility: the named policies live in the non-inline namespace
 * `exception_policies`. Blessed patterns are a block-scope
 * `using namespace cuda::experimental::stf::exception_policies;` at the function that
 * configures sinks, or a namespace alias (`namespace pol = ...::exception_policies;`):
 *
 * @code
 * using namespace cuda::experimental::stf::exception_policies;
 * on_throw(notify & retry * 3 | subst(-1)) << flaky;
 *
 * namespace pol = cuda::experimental::stf::exception_policies;
 * on_throw(pol::subst(0)) << flaky;
 * @endcode
 *
 * @note When querying `noexcept(on_throw(policy) << f)` in a constant expression, pass a
 * location explicitly: nvcc's front-end with a gcc host reports the defaulted
 * `source_location::current()` argument as potentially throwing, tainting the query (the
 * call itself is `noexcept` either way).
 *
 * @param[in] reaction The policy (or a reaction normalized into one), owned if passed an
 *            rvalue and referred to if passed an lvalue.
 * @param[in] loc The location passed to exception hooks.
 * @return A policy object consumed by `operator<<`.
 */
template <class Reaction>
auto on_throw(Reaction&& reaction,
              const ::cuda::std::source_location loc = ::cuda::std::source_location::current()) noexcept
{
  return exception_policies::detail::on_throw_policy{
    exception_policies::detail::normalize(::cuda::std::forward<Reaction>(reaction)), loc};
}

#ifdef ON_THROW
#  error "CUDASTF's scope_guard.cuh defines ON_THROW; rename the prior definition"
#endif
//! @brief Statement-shaped on_throw: ON_THROW(policy-expression) { body };
//! The policy expression is evaluated with `exception_policies` visible, so
//! ON_THROW(notify & retry * 3 | subst(-1)) { return flaky(); }; needs no
//! qualification. Expands to on_throw(...) << a reference-capturing lambda;
//! the call-site location is captured exactly as with plain on_throw. The
//! macro ends at `[&]()`: supply the body type by composition when needed,
//! as in ON_THROW(retry | subst(-1)) -> int { throw failure(); };.
#define ON_THROW(...)                                              \
  ::cuda::experimental::stf::on_throw([&] {                        \
    using namespace ::cuda::experimental::stf::exception_policies; \
    return (VA_ARGS__);                                          \
  }())                                                             \
    << [&]()

#ifdef UNITTESTED_FILE
UNITTEST("nothing")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
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
    return cuda::experimental::stf::exception_policies::abort();
  };
  // A `nothing` expression also supplies one arm of a ternary, the other arm setting the type.
  const auto pick = [](bool ok) -> int {
    return ok ? 42 : cuda::experimental::stf::exception_policies::abort();
  };
  EXPECT(pick(true) == 42);
};

// Negative-compile expectations (do not compile; kept as comments near the code they guard):
//  - abort();                                      // deleted tripwire: qualify exception_policies::abort
//  - on_throw(abort & notify) << [] {};            // "policies after a never-returning policy are unreachable"
//  - on_throw(defer) << [] { return 1; };          // result has no channel (on_success() only)
//  - on_throw(notify) << []() noexcept {};         // existing rule, unchanged message
//  - on_throw(notify & subst(42)) << []() -> int& {...}; // reference result vs owned substitution (existing rule)
//  - on_throw(retry | as_expected) << []() -> int { ... };
//      // conversion failure: an owner's unexpected answer does not convert to the
//      // callable's result; owners belong at the top (or left of &), not in | arms
//  - on_throw(subst(8) | subst(9)) << ...;         // "the left policy never declines; alternatives after it are
//  unreachable"

UNITTEST("on_throw")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
  //! [on_throw]
  // The C library also declares ::abort, so under a using-directive the typed one is picked
  // by name; qualifying every use works as well.
  using cuda::experimental::stf::exception_policies::abort;
  int value = 0;
  on_throw(abort) << [&] {
    value = 42; // would report and abort the application if this code threw
  };
  on_throw(terminate) << [] {};
  on_throw(notify) << [] {}; // would report the exception on stderr and carry on
  const int answer = on_throw(subst(-1)) << [] {
    return 42; // would yield -1 instead if this code threw
  };
  EXPECT(value == 42);
  EXPECT(answer == 42);
  //! [on_throw]

  // A terminating handler declares `nothing` and dies on its own terms; it stays out of the
  // way as long as nothing throws. Raw lambdas of the right shape are policies, no wrapping.
  const auto die = [](const ::std::exception*, ::cuda::std::source_location, auto&) noexcept -> nothing {
    ::std::abort();
  };
  const int untouched = on_throw(die) << [] {
    return 7;
  };
  EXPECT(untouched == 7);

  // Any nullary callable whose declared result is `nothing` works as a terminating action.
  const auto bail = []() noexcept -> nothing {
    ::std::abort();
  };
  const int spared = on_throw(bail) << [] {
    return 9;
  };
  EXPECT(spared == 9);

  // A never-returning reaction goes with a reference result, since it never has to produce
  // one. The referent is static because nvcc reads a return of a by-reference capture as a
  // return of a local.
  static int target = 5;
  int& alias        = on_throw(abort) << []() -> int& {
    return target;
  };
  EXPECT(&alias == &target);

  // A replacement passed as an lvalue outlives the call, so it can stand in for a reference
  // result — bare (adapter) and via subst alike.
  int fallback = 42;
  int& picked  = on_throw(fallback) << []() -> int& {
    return target;
  };
  EXPECT(&picked == &target);

#  if CCCL_HAS_EXCEPTIONS()
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

  // Effects can be plain lambdas: this one counts, then the chain's final element resumes.
  int hits        = 0;
  const auto tick = [&hits](const ::std::exception*, ::cuda::std::source_location, auto&) noexcept {
    ++hits;
  };
  const int ticked = on_throw(noop & tick & ::std::ignore) << []() -> int {
    throw ::std::runtime_error("counted");
  };
  EXPECT(ticked == 0);
  EXPECT(hits == 1);

  // Reporting somewhere other than stderr: notify(stream) is a configured copy of notify.
  ::FILE* const log = ::tmpfile();
  EXPECT(log);
  const auto site  = ::cuda::std::source_location::current();
  const int logged = on_throw(notify(log), site) << []() -> int {
    throw ::std::runtime_error("boom");
  };
  EXPECT(logged == 0);
  // An exception that does not derive from std::exception reaches the handler as nullptr.
  on_throw(notify(log), site) << [] {
    throw 42;
  };
  ::rewind(log);
  char message[1024]{};
  char expected[1024]{};
  EXPECT(::fgets(message, sizeof(message), log));
  ::snprintf(expected,
             sizeof(expected),
             "%s(%u) on_throw violation in %s: boom\n",
             site.file_name(),
             site.line(),
             site.function_name());
  EXPECT(::std::string_view{message} == expected);
  EXPECT(::fgets(message, sizeof(message), log));
  ::snprintf(expected,
             sizeof(expected),
             "%s(%u) on_throw violation in %s: nonstandard exception\n",
             site.file_name(),
             site.line(),
             site.function_name());
  EXPECT(::std::string_view{message} == expected);
  ::fclose(log);

  // The ostream configuration produces the identical report.
  ::std::ostringstream stream_log;
  const int streamed = on_throw(notify(stream_log), site) << []() -> int {
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

  // `defer` captures instead of reacting: empty on success, the exception otherwise, ready
  // for a later rethrow — non-std exceptions included.
  const ::std::exception_ptr clean = on_throw(defer) << [] {};
  EXPECT(!clean);
  const ::std::exception_ptr held = on_throw(defer) << [] {
    throw ::std::runtime_error("deferred");
  };
  EXPECT(!!held);
  bool rethrown = false;
  try
  {
    ::std::rethrow_exception(held);
  }
  catch (const ::std::runtime_error& e)
  {
    rethrown = ::std::string_view{e.what()} == "deferred";
  }
  EXPECT(rethrown);
  const ::std::exception_ptr odd = on_throw(defer) << [] {
    throw 42;
  };
  EXPECT(!!odd);

  // A replacement value stands in for the result, converted to the callable's result type;
  // bare values still work, subst is the documented spelling.
  const int replaced = on_throw(42) << []() -> int {
    throw ::std::runtime_error("replaced");
  };
  EXPECT(replaced == 42);
  const double widened = on_throw(subst(42)) << []() -> double {
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
  const movable moved = on_throw(subst(movable{7})) << []() -> movable {
    throw 42;
  };
  EXPECT(moved.v == 7);
#  endif // CCCL_HAS_EXCEPTIONS()
};

UNITTEST("policy algebra")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
#  if CCCL_HAS_EXCEPTIONS()
  // Vocabulary for observing effect order.
  ::std::string trace;
  const auto mark = [&trace](char c) {
    return [&trace, c](const ::std::exception*, ::cuda::std::source_location, auto&) noexcept {
      trace += c;
    };
  };

  // & sequences left to right; the last element answers.
  trace.clear();
  const int r1 = on_throw(noop & mark('a') & mark('b') & subst(3)) << []() -> int {
    throw ::std::runtime_error("x");
  };
  EXPECT(r1 == 3);
  EXPECT(trace == "ab");

  // & is associative (behaviorally).
  trace.clear();
  const int r2 = on_throw((noop & mark('a') & mark('b')) & subst(3)) << []() -> int {
    throw ::std::runtime_error("x");
  };
  trace += '|';
  const int r3 = on_throw(noop & (mark('a') & (mark('b') & subst(3)))) << []() -> int {
    throw ::std::runtime_error("x");
  };
  EXPECT(r2 == 3);
  EXPECT(r3 == 3);
  EXPECT(trace == "ab|ab");

  // noop is the identity of &.
  trace.clear();
  const int r4 = on_throw(noop & mark('a') & subst(1)) << []() -> int {
    throw ::std::runtime_error("x");
  };
  const int r5 = on_throw(mark('a') & subst(1)) << []() -> int {
    throw ::std::runtime_error("x");
  };
  EXPECT(r4 == 1);
  EXPECT(r5 == 1);
  EXPECT(trace == "aa");

  // | alternation: the left side gets first claim; declining (throwing) passes to the right.
  // rethrow is |'s identity.
  const int r6 = on_throw(rethrow | subst(7)) << []() -> int {
    throw ::std::runtime_error("x");
  };
  EXPECT(r6 == 7);
  const int r7 = on_throw(subst(8) | rethrow) << []() -> int {
    throw ::std::runtime_error("x");
  };
  EXPECT(r7 == 8);

  // catch_only reconstructs the catch ladder: matching type handles, mismatch falls through,
  // non-std exceptions always decline.
  const int r8 = on_throw(catch_only<::std::logic_error>(subst(1)) | subst(2)) << []() -> int {
    throw ::std::logic_error("l");
  };
  EXPECT(r8 == 1);
  const int r9 = on_throw(catch_only<::std::logic_error>(subst(1)) | subst(2)) << []() -> int {
    throw ::std::runtime_error("r");
  };
  EXPECT(r9 == 2);
  const int r10 = on_throw(catch_only<::std::exception>(subst(1)) | subst(2)) << []() -> int {
    throw 42; // reaches the handler as nullptr: catch_only must decline
  };
  EXPECT(r10 == 2);

  // Derived-to-base matching, like a real catch clause.
  const int r11 = on_throw(catch_only<::std::exception>(subst(1)) | subst(2)) << []() -> int {
    throw ::std::runtime_error("derived");
  };
  EXPECT(r11 == 1);

  // Multi-type: either listed exception is claimed; others decline.
  {
    const int a = on_throw(catch_only<::std::logic_error, ::std::overflow_error>(subst(1)) | subst(2)) << []() -> int {
      throw ::std::overflow_error("o");
    };
    EXPECT(a == 1);
    const int b = on_throw(catch_only<::std::logic_error, ::std::overflow_error>(subst(1)) | subst(2)) << []() -> int {
      throw ::std::runtime_error("r");
    };
    EXPECT(b == 2);
  }

  // The correct cascade order -- derived before base -- is legal and behaves.
  {
    const int a = on_throw(catch_only<::std::runtime_error>(subst(1)) | catch_only<::std::exception>(subst(2)))
               << []() -> int {
      throw ::std::runtime_error("r");
    };
    EXPECT(a == 1);
    const int b = on_throw(catch_only<::std::runtime_error>(subst(1)) | catch_only<::std::exception>(subst(2)))
               << []() -> int {
      throw ::std::logic_error("l");
    };
    EXPECT(b == 2);
  }

  // A declining inner policy keeps the right arm live even under a broader left guard: the
  // starved-arm theorem requires a never-declining inner, and this inner declines non-matches.
  {
    const int v = on_throw(catch_only<::std::exception>(catch_only<::std::runtime_error>(subst(1))) | subst(2))
               << []() -> int {
      throw ::std::logic_error("l");
    };
    EXPECT(v == 2);
  }

  // Nonstandard exception types work as guards: matching is by catch-clause rules.
  {
    const int a = on_throw(catch_only<int>(subst(-7)) | subst(0)) << []() -> int {
      throw 42;
    };
    EXPECT(a == -7);
    const int b = on_throw(catch_only<int>(subst(-7)) | subst(0)) << []() -> int {
      throw 3.14;
    };
    EXPECT(b == 0);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - catch_only<::std::exception, ::std::runtime_error>(subst(1));
  //      -> "... the Derived entry is dead (Base already claims it)"
  //  - catch_only<::std::runtime_error, ::std::runtime_error>(subst(1));
  //      -> same (a duplicate subsumes itself)

  // & binds tighter than |, so the ladder below parses as intended without parentheses.
  trace.clear();
  const int r12 = on_throw(catch_only<::std::logic_error>(subst(1)) | mark('n') & subst(2)) << []() -> int {
    throw ::std::runtime_error("r");
  };
  EXPECT(r12 == 2);
  EXPECT(trace == "n");

  // An unanswered chain propagates: retry alone rethrows after exhaustion.
  int attempts = 0;
  bool escaped = false;
  try
  {
    on_throw(retry * 2) << [&] {
      ++attempts;
      throw ::std::runtime_error("always");
    };
  }
  catch (const ::std::runtime_error&)
  {
    escaped = true;
  }
  EXPECT(escaped);
  EXPECT(attempts == 3); // 1 first try + 2 re-invocations

  // retry | terminal: the terminal handles the exhausted failure. Success stops the loop.
  attempts      = 0;
  const int r13 = on_throw(retry * 5 | subst(-1)) << [&]() -> int {
    if (++attempts < 3)
    {
      throw ::std::runtime_error("transient");
    }
    return 99;
  };
  EXPECT(r13 == 99);
  EXPECT(attempts == 3);

  attempts      = 0;
  const int r14 = on_throw(retry * 1 | subst(-1)) << [&]() -> int {
    ++attempts;
    throw ::std::runtime_error("always");
  };
  EXPECT(r14 == -1);
  EXPECT(attempts == 2);

  // noexcept surface: chains of nothrow hooks keep operator<< noexcept; rethrow removes it.
  // The locations are explicit: under nvcc in C++17 mode with a gcc host, evaluating the
  // defaulted source_location::current() argument inside a noexcept operand reads as
  // potentially throwing (the builtin_LINE machinery), which would taint the query with
  // something these assertions do not mean to test.
  static_assert(noexcept(on_throw(notify, ::cuda::std::source_location{}) << ::cuda::std::declval<void (&)()>()),
                "nothrow policy chain must keep the expression noexcept");
  static_assert(!noexcept(on_throw(rethrow, ::cuda::std::source_location{}) << ::cuda::std::declval<void (&)()>()),
                "a throwing policy must surface in the expression's noexcept");

  // Identity elimination is type-level: composing with a neutral element adds no wrapper.
  static_assert(::cuda::std::is_same_v<decltype(noop & subst(1)), decltype(subst(1))>,
                "noop is eliminated on the left");
  static_assert(::cuda::std::is_same_v<decltype(subst(1) & noop), decltype(subst(1))>,
                "noop is eliminated on the right");
  static_assert(::cuda::std::is_same_v<decltype(rethrow | subst(1)), decltype(subst(1))>,
                "rethrow is eliminated on the left");
  static_assert(::cuda::std::is_same_v<decltype(subst(1) | rethrow), decltype(subst(1))>,
                "rethrow is eliminated on the right");

  // noop & abort eliminates to abort itself -- no adapter in the type.
  static_assert(::cuda::std::is_same_v<decltype(noop & exception_policies::abort), abort_t>,
                "abort is a policy; elimination returns it bare");
  {
    using cuda::experimental::stf::exception_policies::abort; // block-scope: hides ::abort
    const int kept = on_throw(noop & abort) << [] {
      return 11;
    };
    EXPECT(kept == 11);
  }
  // (Death paths are untestable here; the report-then-die contract is by inspection.)

  // Behavior after elimination is unchanged (the r6/r7 identity tests above already
  // exercise the runtime side; keep them).

  // Uniform normal forms: identity elimination is idempotent for every operand,
  // including raw callables (which now normalize to a tagged adapter).
  {
    auto raw = [](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      return 5;
    };
    static_assert(::cuda::std::is_same_v<decltype(noop & raw), decltype(noop & (noop & raw))>,
                  "normalization is idempotent: eliminating noop twice adds nothing");
    static_assert(::cuda::std::is_same_v<decltype(noop & raw), decltype((noop & raw) & noop)>,
                  "left and right elimination agree on the normal form");
    const int v = on_throw(noop & raw) << []() -> int {
      throw ::std::runtime_error("x");
    };
    EXPECT(v == 5);
  }

  // Raw-lambda chains headed by noop keep working across the new adapter.
  // (The existing r1..r4 heading-noop tests already cover this; they must
  //  still pass unmodified.)

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(catch_only<::std::exception>(subst(1)) | catch_only<::std::runtime_error>(subst(2))) << ...;
  //      -> "the left catch_only already claims every exception type the right arm lists; ..."
  //  - on_throw(catch_only<::std::exception>(subst(1)) | expecting<::std::runtime_error>) << ...;
  //      -> same (a typed expecting arm is starved the same way)
  //  - on_throw(subst(8) | subst(9)) << []() -> int { throw 1; };
  //      -> "the left policy never declines; alternatives after it are unreachable"
#  endif // CCCL_HAS_EXCEPTIONS()
};

UNITTEST("policy inventory")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
#  if CCCL_HAS_EXCEPTIONS()
  // subst: eager value, lazy callable, and exception-aware callable.
  int lazy_calls = 0;
  const int s1   = on_throw(subst([&lazy_calls] {
                   ++lazy_calls;
                   return 5;
                   }))
                << []() -> int {
    return 1; // success: the lazy fallback must NOT run
  };
  EXPECT(s1 == 1);
  EXPECT(lazy_calls == 0);
  const int s2 = on_throw(subst([&lazy_calls] {
                   ++lazy_calls;
                   return 5;
                 }))
              << []() -> int {
    throw ::std::runtime_error("x");
  };
  EXPECT(s2 == 5);
  EXPECT(lazy_calls == 1);

  const int s3 = on_throw(subst([](const ::std::exception* e, ::cuda::std::source_location, auto&) noexcept {
                   return e ? 1 : 2;
                 }))
              << []() -> int {
    throw 42;
  };
  EXPECT(s3 == 2); // non-std exception: handler sees nullptr

  // defer composes now: report, then capture.
  ::std::ostringstream noted;
  const ::std::exception_ptr np = on_throw(notify(noted) & defer) << [] {
    throw ::std::runtime_error("noted+deferred");
  };
  EXPECT(!!np);
  EXPECT(noted.str().find("noted+deferred") != ::std::string::npos);

  // as_expected: success wraps the value; failure wraps the exception; void works.
  const auto good = on_throw(as_expected) << []() -> int {
    return 5;
  };
  static_assert(::cuda::std::is_same_v<decltype(good), const ::cuda::std::expected<int, ::std::exception_ptr>>,
                "as_expected owns the expression type");
  EXPECT(good.has_value());
  // Dereference rather than .value(): value() would instantiate bad_expected_access, whose
  // inlined exception_ptr destructor trips a spurious gcc 14/15 -O3 maybe-uninitialized in
  // every TU that compiles these tests.
  EXPECT(*good == 5);

  const auto bad = on_throw(as_expected) << []() -> int {
    throw ::std::runtime_error("wrapped");
  };
  EXPECT(!bad.has_value());
  bool wrapped = false;
  try
  {
    ::std::rethrow_exception(bad.error());
  }
  catch (const ::std::runtime_error& e)
  {
    wrapped = ::std::string_view{e.what()} == "wrapped";
  }
  EXPECT(wrapped);

  const auto vgood = on_throw(as_expected) << [] {};
  EXPECT(vgood.has_value());

  // Rightmost success hook wins: defer to the right of as_expected owns the expression.
  const ::std::exception_ptr rm = on_throw(as_expected & defer) << [] {};
  EXPECT(!rm);
#  endif // CCCL_HAS_EXCEPTIONS()
};

UNITTEST("re-running policies")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if CCCL_HAS_EXCEPTIONS()
  // retry | retry: attempts add up (1 initial + 1 + 1).
  {
    int calls    = 0;
    bool escaped = false;
    try
    {
      on_throw(retry | retry) << [&]() -> int {
        ++calls;
        throw ::std::runtime_error("always");
      };
    }
    catch (const ::std::runtime_error&)
    {
      escaped = true;
    }
    EXPECT(escaped);
    EXPECT(calls == 3);
  }

  // Success on a re-attempt returns the callable's result.
  {
    int calls   = 0;
    const int v = on_throw(retry * 3) << [&] {
      if (++calls < 3)
      {
        throw ::std::runtime_error("transient");
      }
      return 42;
    };
    EXPECT(v == 42);
    EXPECT(calls == 3);
  }

  // Void callable: re-attempt success completes the void expression.
  {
    int calls = 0;
    on_throw(retry) << [&] {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
    };
    EXPECT(calls == 2);
  }

  // Effect between attempt groups: the right |-arm's effect fires before its retry.
  {
    int calls = 0;
    int notes = 0;
    auto note = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++notes;
    };
    bool escaped = false;
    try
    {
      on_throw(retry | (note & retry)) << [&]() -> int {
        ++calls;
        throw ::std::runtime_error("always");
      };
    }
    catch (...)
    {
      escaped = true;
    }
    EXPECT(escaped);
    EXPECT(calls == 3); // 1 initial + 1 (left) + 1 (right, after the note)
    EXPECT(notes == 1);
  }

  // Exhausted retry answered by a | fallback.
  {
    int calls   = 0;
    const int v = on_throw(retry * 2 | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }

  // A plain ignore-arm after retry resumes with a default-constructed result.
  {
    int calls   = 0;
    const int v = on_throw(retry | ::std::ignore) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("always");
    };
    EXPECT(v == 0);
    EXPECT(calls == 2);
  }

  // catch_only restricts what gets re-run: wrong type declines without re-running.
  {
    int calls   = 0;
    const int v = on_throw(catch_only<::std::logic_error>(retry * 5) | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("not a logic_error");
    };
    EXPECT(v == -1);
    EXPECT(calls == 1); // no re-runs: catch_only declined before retry saw it
  }
  {
    int calls   = 0;
    const int v = on_throw(catch_only<::std::logic_error>(retry * 2) | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::logic_error("is one");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3); // re-run twice, exhausted, then the fallback
  }

  // RULING: the chain's on_success fires on re-attempt success.
  {
    int calls    = 0;
    const auto r = on_throw(as_expected & retry) << [&] {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
      return 7;
    };
    EXPECT(r.has_value());
    EXPECT(*r == 7);
    EXPECT(calls == 2);
  }
  {
    int calls     = 0;
    const auto ep = on_throw(defer & retry) << [&] {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
    };
    EXPECT(!ep); // empty: the re-attempt succeeded, so on_success() supplied the value
    EXPECT(calls == 2);
  }

  // The uniform discard law: & throws away non-final answers, even a re-run's.
  {
    int calls   = 0;
    const int v = on_throw(retry & subst(-1)) << [&]() -> int {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
      return 99; // the re-run succeeds...
    };
    EXPECT(v == -1); // ...and & discards its answer; subst answers. Legal, documented, weird.
    EXPECT(calls == 2);
  }

  // A raw 3-arg lambda is a policy and may re-run.
  {
    int calls   = 0;
    const int v = on_throw([](const ::std::exception*, auto, auto& fn) {
                    return fn();
                  })
               << [&]() -> int {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
      return 5;
    };
    EXPECT(v == 5);
    EXPECT(calls == 2);
  }

  // References survive re-running: the re-run hands back the same object, not a copy
  // (this is why the hooks are decltype(auto), not auto).
  {
    static int obj = 5;
    int calls      = 0;
    int& r         = on_throw(retry) << [&]() -> int& {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
      return obj;
    };
    EXPECT(&r == &obj);
    EXPECT(calls == 2);
  }

  // noexcept surface: a re-running chain is never noexcept (explicit location; see
  // the existing comment about nvcc + gcc host and defaulted current()).
  static_assert(!noexcept(on_throw(retry, ::cuda::std::source_location{}) << ::cuda::std::declval<int (&)()>()),
                "a re-running reaction can always decline");

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(subst(1) | retry) << ...;
  //      -> "the left policy never declines; ..." (existing theorem, unchanged)
  //  - on_throw(retry | as_expected) << []() -> int { ... };
  //      -> conversion failure: an owner's unexpected answer does not convert to the
  //         callable's result; owners belong at the top (or left of &), not in | arms
#  endif // CCCL_HAS_EXCEPTIONS()
};

// Helper exception types for the tests below, at namespace scope on purpose: nvcc <= 12.9 in
// C++20 mode infers host__ device__ for a local class's special members inside an extended
// lambda, and the inherited std::runtime_error constructor is host-only (error #20011).
struct ut_derived_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};
struct ut_my_error
{
  int code;
};
struct ut_poly_base
{
  virtual ~ut_poly_base() = default;
};
struct ut_poly_derived : ut_poly_base
{};
struct ut_low_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};
struct ut_high_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};

UNITTEST("expecting")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if CCCL_HAS_EXCEPTIONS()
  // Exact match: the exception object lands by value.
  {
    const auto r = on_throw(expecting<::std::runtime_error>) << []() -> int {
      throw ::std::runtime_error("boom");
    };
    static_assert(::cuda::std::is_same_v<decltype(r), const ::cuda::std::expected<int, ::std::runtime_error>>,
                  "the error slot is the exception type itself, by value");
    EXPECT(!r.has_value());
    EXPECT(::std::string{r.error().what()} == "boom");
  }
  {
    const auto r = on_throw(expecting<::std::runtime_error>) << [] {
      return 42;
    };
    EXPECT(r.has_value());
    EXPECT(*r == 42);
  }

  // A DERIVATIVE declines (no slicing): the full dynamic type survives.
  {
    bool escaped = false;
    try
    {
      on_throw(expecting<::std::runtime_error>) << [&]() -> int {
        throw ut_derived_error{"sliced?"};
      };
    }
    catch (const ut_derived_error&)
    {
      escaped = true;
    }
    EXPECT(escaped);
  }

  // Unrelated exception: declines to the next | arm; the arm's value converts in engaged.
  {
    const auto r = on_throw(expecting<::std::logic_error> | subst(-1)) << []() -> int {
      throw ::std::runtime_error("not a logic_error");
    };
    EXPECT(r.has_value());
    EXPECT(*r == -1);
  }

  // A non-std exception (null hook pointer) declines.
  {
    bool escaped = false;
    try
    {
      on_throw(expecting<::std::runtime_error>) << [&]() -> int {
        throw 42;
      };
    }
    catch (int)
    {
      escaped = true;
    }
    EXPECT(escaped);
  }

  // The exception_ptr specialization is the old as_expected; the alias holds.
  static_assert(::cuda::std::is_same_v<decltype(as_expected), const expecting_t<::std::exception_ptr>>,
                "as_expected is the baseline instance of expecting");
  {
    const auto r = on_throw(expecting<::std::exception_ptr>) << []() -> int {
      throw 42; // even a non-std exception is captured, not declined
    };
    EXPECT(!r.has_value());
    EXPECT(!!r.error());
  }

  // Void callable through the typed form.
  {
    int calls    = 0;
    const auto r = on_throw(expecting<::std::runtime_error>) << [&] {
      ++calls;
    };
    EXPECT(r.has_value());
    EXPECT(calls == 1);
  }

  // Any catchable type works as the error slot -- the former negative-compile case is a feature.
  {
    const auto r = on_throw(expecting<int>) << []() -> double {
      throw 42;
    };
    EXPECT(!r.has_value());
    EXPECT(r.error() == 42);
  }
  {
    const auto r = on_throw(expecting<ut_my_error>) << []() -> int {
      throw ut_my_error{7};
    };
    EXPECT(!r.has_value());
    EXPECT(r.error().code == 7);
  }
  // A nonstandard polymorphic hierarchy still gets exact matching (derivative declines).
  {
    bool escaped = false;
    try
    {
      on_throw(expecting<ut_poly_base>) << []() -> int {
        throw ut_poly_derived{};
      };
    }
    catch (const ut_poly_derived&)
    {
      escaped = true;
    }
    EXPECT(escaped);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(expecting<::std::exception_ptr> | subst(0)) << ...;
  //      -> "the left policy never declines; ..." (the total form can't head a ladder)
#  endif // CCCL_HAS_EXCEPTIONS()
};

UNITTEST("guard translate delay backoff remember")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if CCCL_HAS_EXCEPTIONS()
  // A true guard contributes an effect-only answer; a false guard declines its whole sequence.
  {
    const auto is_low = [](const ::std::exception* exception) {
      return exception && dynamic_cast<const ut_low_error*>(exception);
    };
    const int claimed = on_throw(when(is_low, subst(1)) | subst(2)) << []() -> int {
      throw ut_low_error("low");
    };
    const int declined = on_throw(when(is_low, subst(1)) | subst(2)) << []() -> int {
      throw ut_high_error("high");
    };
    EXPECT(claimed == 1);
    EXPECT(declined == 2);
  }
  {
    const int claimed =
      on_throw(when(
                 [](const ::std::exception* exception) {
                   return !exception;
                 },
                 subst(5))
               | subst(6))
      << []() -> int {
      throw 42;
    };
    EXPECT(claimed == 5);
  }
  {
    const int accepted =
      on_throw(guard([] {
                 return true;
               })
               & subst(3))
      << []() -> int {
      throw ut_low_error("low");
    };
    const int declined =
      on_throw((guard([] {
                  return false;
                })
                & subst(3))
               | subst(4))
      << []() -> int {
      throw ut_low_error("low");
    };
    EXPECT(accepted == 3);
    EXPECT(declined == 4);
  }

  // A translator's thrown exception is re-observed by the next typed arm.
  {
    const int v = on_throw(catch_only<ut_low_error>(translate([](const ::std::exception*) {
                             return ut_high_error{"context"};
                           }))
                           | catch_only<ut_high_error>(subst(1)))
               << []() -> int {
      throw ut_low_error("cause");
    };
    EXPECT(v == 1);
  }

  // Nest preserves the original exception as the translated exception's cause.
  {
    bool saw_high = false;
    bool saw_low  = false;
    try
    {
      on_throw(nest(ut_high_error{"context"})) << [] {
        throw ut_low_error("cause");
      };
    }
    catch (const ut_high_error& exception)
    {
      saw_high = true;
      try
      {
        ::std::rethrow_if_nested(exception);
      }
      catch (const ut_low_error&)
      {
        saw_low = true;
      }
    }
    EXPECT(saw_high);
    EXPECT(saw_low);
  }

  // Delay composes before each retry; test attempts rather than elapsed wall time.
  {
    int calls   = 0;
    const int v = on_throw((delay(::std::chrono::milliseconds{1}) & retry) * 2 | subst(-1)) << [&]() -> int {
      ++calls;
      throw ut_low_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }

  // Backoff owns its retry loop: exhaustion declines, while an early success answers.
  {
    int calls   = 0;
    const int v = on_throw(backoff(2, ::std::chrono::milliseconds{1}) | subst(-1)) << [&]() -> int {
      ++calls;
      throw ut_low_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }
  {
    int calls   = 0;
    const int v = on_throw(backoff(2, ::std::chrono::milliseconds{1})) << [&]() -> int {
      if (++calls == 1)
      {
        throw ut_low_error("once");
      }
      return 8;
    };
    EXPECT(v == 8);
    EXPECT(calls == 2);
  }
  {
    int calls = 0;
    on_throw(backoff(2, ::std::chrono::milliseconds{1})) << [&] {
      if (++calls == 1)
      {
        throw ut_low_error("once");
      }
    };
    EXPECT(calls == 2);
  }

  // Remember observes successes and substitutes the latest one after a failure.
  {
    int last      = 1;
    const int got = on_throw(remember(last)) << [] {
      return 7;
    };
    EXPECT(got == 7);
    EXPECT(last == 7);

    const int stale = on_throw(remember(last)) << []() -> int {
      throw ut_low_error("offline");
    };
    EXPECT(stale == 7);

    const int fresh = ON_THROW(remember(last))
    {
      return 9;
    };
    const int served = ON_THROW(remember(last))->int
    {
      throw ut_low_error("offline");
    };
    EXPECT(fresh == 9);
    EXPECT(last == 9);
    EXPECT(served == 9);
  }
  {
    int last = 0;
    // static: nvcc 12.0's cudafe flags returning a by-ref-captured local as
    // "returning reference to local variable" (#836, promoted); the capture
    // is valid, the old analysis just cannot see through it.
    static int source = 11;
    int& fresh        = on_throw(remember(last)) << [&]() -> int& {
      return source;
    };
    int& stale = on_throw(remember(last)) << []() -> int& {
      throw ut_low_error("offline");
    };
    EXPECT(&fresh == &source);
    EXPECT(last == 11);
    EXPECT(&stale == &last);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(remember(value) | subst(0)) << ...;
  //      -> "the left policy never declines; alternatives after it are unreachable"
  //  - on_throw(translate(fn) & subst(0)) << ...;
  //      -> "policies after a never-returning policy are unreachable"
  //  - remember(42);
  //      -> remember requires an lvalue to hold by reference
#  endif // CCCL_HAS_EXCEPTIONS()
};

UNITTEST("repetition")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
  static_assert(::cuda::std::is_empty_v<retry_t>, "retry is the stateless unit re-attempt");

#  if CCCL_HAS_EXCEPTIONS()
  // retry * n: 1 + n attempts, then the last failure propagates.
  {
    int calls    = 0;
    bool escaped = false;
    try
    {
      on_throw(retry * 3) << [&]() -> int {
        ++calls;
        throw ::std::runtime_error("always");
      };
    }
    catch (const ::std::runtime_error&)
    {
      escaped = true;
    }
    EXPECT(escaped);
    EXPECT(calls == 4);
  }

  // p * 0 is rethrow: no re-attempts, immediate decline.
  {
    int calls    = 0;
    bool escaped = false;
    try
    {
      on_throw(retry * 0) << [&]() -> int {
        ++calls;
        throw ::std::runtime_error("once");
      };
    }
    catch (...)
    {
      escaped = true;
    }
    EXPECT(escaped);
    EXPECT(calls == 1);
  }

  // The flagship: (effect & retry) * 3 fires the effect before EACH re-attempt.
  {
    int calls = 0;
    int notes = 0;
    auto note = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++notes;
    };
    const int v = on_throw((note & retry) * 3 | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 4);
    EXPECT(notes == 3);
  }

  // Precedence contrast: notify-effect once, then the re-attempts.
  {
    int calls = 0;
    int notes = 0;
    auto note = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++notes;
    };
    const int v = on_throw(note & retry * 3 | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 4);
    EXPECT(notes == 1);
  }

  // Success mid-repetition returns the callable's result (and the success channel).
  {
    int calls    = 0;
    const auto r = on_throw((as_expected & retry) * 3) << [&] {
      if (++calls < 3)
      {
        throw ::std::runtime_error("transient");
      }
      return 7;
    };
    EXPECT(r.has_value());
    EXPECT(*r == 7);
    EXPECT(calls == 3);
  }

  // The law p*(m+n) == p*m | p*n, observed through attempt counts.
  {
    int calls    = 0;
    bool escaped = false;
    try
    {
      on_throw(retry * 1 | retry * 2) << [&]() -> int {
        ++calls;
        throw ::std::runtime_error("always");
      };
    }
    catch (...)
    {
      escaped = true;
    }
    EXPECT(escaped);
    EXPECT(calls == 4); // same as retry * 3
  }

  // Commuted form.
  {
    int calls   = 0;
    const int v = on_throw(2 * retry | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }

  // A plain declining arm repeats too: catch_only guards every iteration.
  {
    int calls   = 0;
    const int v = on_throw(catch_only<::std::logic_error>(retry) * 5 | subst(-1)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("not a logic_error");
    };
    EXPECT(v == -1);
    EXPECT(calls == 1); // declined on type before any re-run, every iteration vacuous
  }

  // Repetition preserves reference results too (the | walk is decltype(auto) throughout).
  {
    static int obj = 9;
    int calls      = 0;
    int& r         = on_throw(retry * 2) << [&]() -> int& {
      if (++calls < 3)
      {
        throw ::std::runtime_error("transient");
      }
      return obj;
    };
    EXPECT(&r == &obj);
    EXPECT(calls == 3);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(subst(1) * 3) << ...;
  //      -> "the repeated policy never declines; repetitions after the first are unreachable"
#  endif // CCCL_HAS_EXCEPTIONS()
};

UNITTEST("ON_THROW macro")
{
  using namespace cuda::experimental::stf; // deliberately NOT exception_policies:
                                           // the macro must supply the vocabulary
#  if CCCL_HAS_EXCEPTIONS()
  {
    int calls   = 0;
    const int v = ON_THROW(retry * 2 | subst(-1))->int
    {
      ++calls;
      throw ::std::runtime_error("always"); // every path throws: the arrow supplies the type
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }
  {
    const int v = ON_THROW(subst(7))
    {
      return 1;
    };
    EXPECT(v == 1);
  }
  // Reference result via the composed arrow.
  {
    static int obj = 3;
    int& r         = ON_THROW(subst(obj))->int& // lvalue substitution can serve a reference
    {
      if (obj == 3)
      {
        throw ::std::runtime_error("x");
      }
      return obj;
    };
    EXPECT(&r == &obj);
  }
#  endif // CCCL_HAS_EXCEPTIONS()
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

#ifndef CCCL_DOXYGEN_INVOKED // Do not document
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
  on_throw(exception_policies::abort, loc) << f;
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
#endif // !CCCL_DOXYGEN_INVOKED
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
