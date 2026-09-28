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
#include <cuda/std/type_traits/is_arithmetic.h>
#include <cuda/std/type_traits/is_base_of.h>
#include <cuda/std/type_traits/is_convertible.h>
#include <cuda/std/type_traits/is_default_constructible.h>
#include <cuda/std/type_traits/is_enum.h>
#include <cuda/std/type_traits/is_floating_point.h>
#include <cuda/std/type_traits/is_integral.h>
#include <cuda/std/type_traits/is_reference.h>
#include <cuda/std/type_traits/is_same.h>
#include <cuda/std/type_traits/is_unsigned.h>
#include <cuda/std/type_traits/is_valid_expansion.h>
#include <cuda/std/type_traits/is_void.h>
#include <cuda/std/type_traits/remove_cvref.h>
#include <cuda/std/type_traits/type_identity.h>
#include <cuda/std/type_traits/underlying_type.h>
#include <cuda/std/utility/declval.h>
#include <cuda/std/utility/forward.h>
#include <cuda/std/utility/move.h>

#include <cuda/experimental/stf/utility/source_location.cuh>
#include <cuda/experimental/stf/utility/traits.cuh>
#include <cuda/experimental/stf/utility/unittest.cuh>

#include <any>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <utility>

#ifdef UNITTESTED_FILE
#  include <sstream>
#  include <string>
#endif // UNITTESTED_FILE

namespace cuda::experimental::stf
{
/**
 * @brief The bottom type: a type with no values, convertible to every type.
 *
 * A callable that declares `nullval` as its return type promises in the type system that it
 * never returns normally: keeping the promise any other way would require materializing a value
 * of a type that has none. `[[noreturn]]` makes the same promise to the optimizer, but not
 * reliably to overload resolution; a `nullval` result states it as a fact of the type, visible
 * to metaprogramming and impossible to fake.
 *
 * The conversion operator lets a `nullval` expression appear wherever a value of any type is
 * expected, references included: a never-returning call may be `return`ed from a function of
 * any result type, or supply one arm of a ternary whose other arm produces the legitimate
 * value. The operator can never run -- running it would
 * require an object that cannot exist -- so its body exists to satisfy the compiler, not to
 * execute.
 */
struct nullval final
{
  nullval()                          = delete;
  nullval(const nullval&)            = delete;
  nullval& operator=(const nullval&) = delete;

  // Two operators, because deduction for conversion functions strips the reference off the
  // target before matching: the rvalue one serves values and rvalue references, the lvalue one
  // serves lvalue references. A value target sees both and prefers the rvalue binding, so the
  // pair is not ambiguous. The bodies are unreachable rather than aborting: every `nullval`
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
 * `on_throw(abort) << callable`. The policy acts through its hook only; it has no bare-call
 * form, so `exception_policies::abort()` as a plain statement does not compile. Ending a
 * program directly remains `std::abort()`.
 *
 * Inside `exception_policies`, plain `abort` finds this object before the C library's function.
 * Code that sees both through using-directives gets an ambiguity error rather than a silent
 * pick, and disambiguates with a using-declaration:
 * `using cuda::experimental::stf::exception_policies::abort;`. A block-scope using-declaration
 * still hides `::abort`.
 *
 * `notify & abort` reports twice (documented). `abort | p` is a dead-| error (hook is
 * noexcept); `abort & p` is a dead-& error (answers `nullval`).
 */
struct abort_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  //! @brief The exception hook: report, then die.
  template <class Fn>
  [[noreturn]] nullval
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

  template <class Fn>
  [[noreturn]] nullval
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
 * @brief Capturing policy: on a throw, answers with the active `std::exception_ptr`, ready for
 * storage and a later `std::rethrow_exception`. This is the policy for boundaries that must
 * not unwind but cannot decide either -- the exception's fate is somebody else's, later.
 *
 * The callable owns the expression type, so it must return `std::exception_ptr` on success:
 * `return std::exception_ptr();`. A throw-only callable spells that return type explicitly.
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
};
inline constexpr defer_t defer{};

/**
 * @brief The identity element of `|` and the decline primitive: re-throws the in-flight
 * exception from inside the catch. Its answer type is `nullval`, so it never has to produce a
 * value; being non-`noexcept` is how it declines, handing the exception to the next `|` arm or
 * letting it propagate.
 */
struct rethrow_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  template <class Fn>
  [[noreturn]] nullval operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const
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

  V v_;

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
  // not to a copy (the ignore branch deduces the object type and returns a copy of the tag).
  template <class Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, Fn& fn)
  {
    if constexpr (::cuda::std::is_void_v<decltype(fn())>)
    {
      fn(); // a throw here IS the decline
      return ::std::ignore; // resume: the void expression is complete
    }
    else
    {
      return fn();
    }
  }
};
inline constexpr retry_t retry{};

namespace detail
{
template <class>
inline constexpr bool is_expected = false;

template <class T, class E>
inline constexpr bool is_expected<::cuda::std::expected<T, E>> = true;

template <class>
struct expected_error;

template <class T, class E>
struct expected_error<::cuda::std::expected<T, E>>
{
  using type = E;
};
} // namespace detail

/**
 * @brief Converts an exception into the error channel of the callable's `cuda::std::expected`
 * result.
 *
 * The callable must return a `cuda::std::expected<T, E>` specialization. `E` is deduced from
 * that return type and constructed first from `std::exception_ptr`, when possible, otherwise
 * from the funneled `const std::exception&`. A nonstandard exception declines when only the
 * latter construction is available.
 */
struct as_expected_t
{
  using exception_sink_tag = void;

  template <class Fn, class Raw = decltype(::cuda::std::declval<Fn&>()())>
  auto operator()(const ::std::exception* exception, const ::cuda::std::source_location, Fn&) const -> Raw
  {
    // nvcc instantiates this body when forming the callable-independent presence probe
    // (`void (&)()`). Keep that archetype admissible; the assert still fires at a real
    // composition site (a user void-lambda is a distinct type, not `void()`).
    if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<Fn>, void()>)
    {
      CCCL_UNREACHABLE();
    }
    else
    {
      using Expected = ::cuda::std::remove_cvref_t<Raw>;
      static_assert(detail::is_expected<Expected>,
                    "as_expected requires the callable to return a cuda::std::expected instantiation");

      if constexpr (detail::is_expected<Expected>)
      {
        using E = typename detail::expected_error<Expected>::type;
        if constexpr (::cuda::std::is_constructible_v<E, ::std::exception_ptr>)
        {
          return Raw{::cuda::std::unexpect, E(::std::current_exception())};
        }
        else if constexpr (::cuda::std::is_constructible_v<E, const ::std::exception&>)
        {
          if (exception)
          {
            return Raw{::cuda::std::unexpect, E(*exception)};
          }
          throw; // nonstandard exception, no lossless construction rung: decline
        }
        else
        {
          static_assert(
            ::cuda::std::is_constructible_v<E, ::std::exception_ptr>
              || ::cuda::std::is_constructible_v<E, const ::std::exception&>,
            "as_expected requires the error type constructible from exception_ptr or const std::exception&");
        }
      }
      CCCL_UNREACHABLE();
    }
  }
};
inline constexpr as_expected_t as_expected{};

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
struct when_t
{
  using exception_sink_tag = void;
  Pred pred_;

  template <class Fn>
  // maybe_unused: when the predicate is nullary, only the discarded constexpr
  // branch reads exception; gcc 9 reports it as set-but-unused.
  void operator()([[maybe_unused]] const ::std::exception* exception, const ::cuda::std::source_location, Fn&)
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

/**
 * @brief Boundary translation: catches a `From` (catch-clause rules: same or publicly
 * derived) and throws a `To` -- constructed from the caught `From` when such a constructor
 * exists, default-constructed otherwise. Anything that is not a `From` declines untouched,
 * so `translate<low, high> | ...` ladders compose; a following arm sees the `To`.
 */
template <class From, class To>
struct translate_t
{
  using exception_sink_tag = void;

  template <class Fn>
  [[noreturn]] nullval operator()(const ::std::exception* e, const ::cuda::std::source_location, Fn&) const
  {
    if (e)
    {
      // The funnel pointer decides for std-derived exceptions, no rethrow needed.
      if (const auto* from = dynamic_cast<const From*>(e))
      {
        throw_translated(*from);
      }
      throw; // decline: a std exception that is not a From
    }
    // A non-std exception: re-observe at From.
    CCCL_TRY
    {
      throw;
    }
    CCCL_CATCH (const From& from)
    {
      throw_translated(from);
    }
    CCCL_CATCH_FALLTHROUGH // decline: not a From either
    CCCL_UNREACHABLE();
  }

private:
  [[noreturn]] static void throw_translated(const From& from)
  {
    if constexpr (::cuda::std::is_constructible_v<To, const From&>)
    {
      throw To(from);
    }
    else if constexpr (::cuda::std::is_base_of_v<::std::exception, From>
                       && ::cuda::std::is_constructible_v<To, const char*>)
    {
      throw To(from.what()); // carry the message across the translation
    }
    else if constexpr (::cuda::std::is_default_constructible_v<To>)
    {
      throw To{};
    }
    else
    {
      static_assert(!::cuda::std::is_same_v<From, From>,
                    "translate<From, To>: To must be constructible from const From&, from "
                    "From::what(), or default-constructible");
    }
  }
};

//! @brief Translates: catches a `From` (catch-clause rules), throws a `To` -- constructed from
//! the caught `From` when possible, default-constructed otherwise. Anything else declines.
template <class From, class To>
inline constexpr translate_t<From, To> translate{};

/**
 * @brief Throws a stored exception with the active exception nested as its cause.
 */
template <class E>
struct nest_t
{
  using exception_sink_tag = void;
  E exception_;

  static_assert(::cuda::std::is_copy_constructible_v<E>, "nest(e) requires a copyable exception object");

  template <class Fn>
  [[noreturn]] nullval operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&)
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
  Duration duration_;

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
    // maybe_unused: referenced only inside CCCL_CATCH_ALL, which the device pass
    // expands to a discarded branch; CTK <= 12.9's cudafe then reports #177.
    [[maybe_unused]] const auto cap = base * 64;
    auto sleep                      = base;
    auto state = static_cast<unsigned long long>(::std::chrono::steady_clock::now().time_since_epoch().count());
    if (state == 0)
    {
      state = 1;
    }

    // maybe_unused: like cap above, left is referenced only inside
    // CCCL_CATCH_ALL, so CTK <= 12.9's cudafe reports #177 without it.
    for ([[maybe_unused]] int left = n_;;)
    {
      ::std::this_thread::sleep_for(::std::chrono::milliseconds{sleep});
      CCCL_TRY
      {
        if constexpr (::cuda::std::is_void_v<decltype(fn())>)
        {
          fn();
          return ::std::ignore;
        }
        else
        {
          return fn();
        }
      }
      CCCL_CATCH_ALL
      {
        if (--left == 0)
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
 * The cell is designated by pointer -- a raw `T*` for a caller-owned variable, or a
 * `shared_ptr<T>` when the policy should share ownership; the two are distinguished
 * statically by overload. Success updates the cell and passes the result through; failure
 * substitutes the stored value (an lvalue: it can serve reference results).
 */
template <class Ptr>
struct remember_t
{
  using exception_sink_tag = void;
  Ptr cell_; // a raw pointer or a shared_ptr; *cell_ is the last-known-good value

  template <class R>
  ::cuda::std::conditional_t<::cuda::std::is_lvalue_reference_v<R&&>, R&&, ::cuda::std::remove_cvref_t<R>>
  on_success(R&& result)
  {
    *cell_ = result;
    return ::cuda::std::forward<R>(result);
  }

  template <class Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const noexcept
  {
    return *cell_;
  }
};

//! @brief Creates a last-known-good policy over a caller-owned cell.
template <class T>
auto remember(T* cell)
{
  CCCL_ASSERT(cell, "remember requires a non-null cell");
  return remember_t<T*>{cell};
}

//! @brief Creates a last-known-good policy that shares ownership of its cell.
template <class T>
auto remember(::std::shared_ptr<T> cell)
{
  CCCL_ASSERT(cell, "remember requires a non-null cell");
  return remember_t<::std::shared_ptr<T>>{::cuda::std::move(cell)};
}

//! @brief Thrown by @ref circuit_breaker_t "circuit_breaker" to refuse an attempt while the
//! circuit is open. It escapes the whole guarded expression: the policy that raises it never
//! handles it.
struct circuit_open : ::std::runtime_error
{
  circuit_open()
      : ::std::runtime_error("circuit breaker open: the failure budget is spent")
  {}
};

/**
 * @brief Counter-based circuit breaker over a caller-owned failure budget.
 *
 * The budget is a `shared_ptr<int>` holding the number of failures the circuit absorbs before
 * opening. Each exception decrements it (the hook is an effect and answers `std::ignore`); a
 * success restores it to the value it held at creation. Once the budget is spent, the entry
 * gate refuses further attempts by throwing @ref circuit_open before the callable runs: the
 * failing dependency gets quiet instead of hammering, and callers fail fast instead of piling
 * up. The budget is shared and caller-owned, so several call sites may gate on one circuit,
 * and writing to the `int` administers the breaker externally (a monitor may re-close the
 * circuit by refilling it).
 *
 * Use as an `&` arm ahead of the recovery, e.g.
 * `circuit_breaker(budget) & retry * 2 | notify & subst(fallback)`.
 */
struct circuit_breaker_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  ::std::shared_ptr<int> budget_;
  int initial_;

  //! @brief The entry gate: refuses the attempt once the budget is spent.
  void on_enter() const
  {
    if (*budget_ <= 0)
    {
      throw circuit_open{};
    }
  }

  //! @brief The exception hook: record the failure, answer as an effect.
  template <class Fn>
  decltype(::std::ignore)
  operator()(const ::std::exception*, const ::cuda::std::source_location, Fn&) const noexcept
  {
    --*budget_;
    return ::std::ignore;
  }

  //! @brief Success restores the budget to its creation-time value.
  template <class R>
  ::cuda::std::conditional_t<::cuda::std::is_lvalue_reference_v<R&&>, R&&, ::cuda::std::remove_cvref_t<R>>
  on_success(R&& result) const
  {
    *budget_ = initial_;
    return ::cuda::std::forward<R>(result);
  }

  void on_success() const
  {
    *budget_ = initial_;
  }
};

//! @brief Creates a counter-based circuit breaker; see @ref circuit_breaker_t. The initial
//! `*budget` is the failure allowance restored on success.
inline circuit_breaker_t circuit_breaker(::std::shared_ptr<int> budget)
{
  CCCL_ASSERT(budget, "circuit_breaker requires a non-null budget");
  const int initial = *budget;
  return circuit_breaker_t{::std::move(budget), initial};
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

// Capability 2c: the entry hook `p.on_enter()`, run once per attempt expression, before the
// callable and outside the policy's own catch. It answers nothing: it either admits the
// attempt or refuses it by throwing.
template <class P>
using on_enter_of = decltype(::cuda::std::declval<P&>().on_enter());

template <class P>
inline constexpr bool has_on_enter =
  ::cuda::std::IsValidExpansion<on_enter_of, ::cuda::std::remove_reference_t<P>>::value;

// Nothrow-ness of the entry hook, vacuously true when absent (two-step form: `&&` does not
// short-circuit template instantiation).
template <class P, bool = has_on_enter<P>>
inline constexpr bool on_enter_nothrow_v = true;

template <class P>
inline constexpr bool on_enter_nothrow_v<P, true> =
  noexcept(::cuda::std::declval<::cuda::std::remove_reference_t<P>&>().on_enter());

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

// Whether a policy's answer type is `nullval` -- it never returns from the exception path. The
// two-step form keeps `hook_answer_t` from being named for a hookless policy: `&&` does not
// short-circuit template instantiation, so the answer is probed only in the `true` partial.
template <bool HasHook, class P, class Fn>
inline constexpr bool answers_nothing_impl = false;

template <class P, class Fn>
inline constexpr bool answers_nothing_impl<true, P, Fn> =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<hook_answer_t<P, Fn>>, nullval>;

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
// `as_policy`, `policy_pow`): both arities delegate to the wrapped policy. The outer
// `operator<<` enforces that the forwarded answer preserves the callable's expression type.
template <class P>
struct forwards_success
{
  P p_;

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

  template <class Self = P, ::cuda::std::enable_if_t<has_on_enter<Self>, int> = 0>
  void on_enter() noexcept(on_enter_nothrow_v<P>)
  {
    p_.on_enter();
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
  static bool matches_active(const ::std::exception* e)
  {
    // Fast path: for a class target and a std-derived active exception, the funnel pointer
    // decides via dynamic_cast -- no rethrow. (dynamic_cast agrees with catch matching:
    // ambiguous or non-public bases yield null, and a catch clause would not match either.)
    if constexpr (::cuda::std::is_class_v<E0>)
    {
      if (e)
      {
        if (dynamic_cast<const E0*>(e))
        {
          return true;
        }
        if constexpr (sizeof...(Rest) > 0)
        {
          return matches_active<Rest...>(e);
        }
        else
        {
          return false;
        }
      }
    }
    // Slow path: a non-class target, or a non-std active exception -- re-observe.
    CCCL_TRY
    {
      throw;
    }
    CCCL_CATCH ([[maybe_unused]] const E0& match)
    {
      return true;
    }
    CCCL_CATCH_ALL
    {
      if constexpr (sizeof...(Rest) > 0)
      {
        return matches_active<Rest...>(e);
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
    if (matches_active<Es...>(exception))
    {
      // A matching non-std exception still reaches `P` as a null pointer, per the funnel.
      return this->p_(exception, loc, fn);
    }
    throw; // decline: no listed type claims the active exception
  }
};

// Intra-pack duplicates are dead for exact matching; cone relations are fine (Base and
// Derived may both be listed, each matching only its own dynamic type).
template <class...>
inline constexpr bool catch_exactly_pack_ok = true;

template <class Head, class... Tail>
inline constexpr bool catch_exactly_pack_ok<Head, Tail...> =
  (!::cuda::std::is_same_v<Head, Tail> && ...) && catch_exactly_pack_ok<Tail...>;

// Is `B` textually one of `As...`? The exact-guard analogue of `claimed_by_any`.
template <class B, class... As>
inline constexpr bool listed_exactly = (::cuda::std::is_same_v<As, B> || ...);

// `catch_exactly<E1, E2, ...>(p)`: run `p`'s exception path when the active exception's
// DYNAMIC type is exactly one of the listed types, else decline by rethrowing. Monomorphic
// where `catch_only` is polymorphic: derived types do not match, so a handler accepts a type
// without inheriting its cone. Matching reads typeid through the std::exception funnel, so
// listed types must derive std::exception (enforced by the factory); a non-std active
// exception (null funnel) always declines.
template <class P, class... Es>
struct catch_exactly_t : forwards_success<P>
{
  using exception_sink_tag = void;

  template <class Fn, class Self = P, ::cuda::std::enable_if_t<has_exception_hook<Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
  {
    if (exception != nullptr && ((typeid(*exception) == typeid(Es)) || ...))
    {
      return this->p_(exception, loc, fn);
    }
    throw; // decline: the active exception's dynamic type is not listed
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

// Success-hook forwarding shared by the `&` and `|` composites: rightmost-wins. The outer
// `operator<<` enforces that the selected hook preserves the callable's expression type. The
// exception hook -- where the two composites differ -- lives in the derived types.
template <class L, class R>
struct composite_hooks
{
  L l_;
  R r_;

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

  // Entry gates run left to right: each side may refuse the attempt before it starts.
  template <class LL                                                                 = L,
            class RR                                                                 = R,
            ::cuda::std::enable_if_t<has_on_enter<LL> || has_on_enter<RR>, int> = 0>
  void on_enter() noexcept(on_enter_nothrow_v<L> && on_enter_nothrow_v<R>)
  {
    if constexpr (has_on_enter<L>)
    {
      l_.on_enter();
    }
    if constexpr (has_on_enter<R>)
    {
      r_.on_enter();
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
      static_cast<void>(this->l_(exception, loc, fn)); // non-final answers are discarded
    }
    if constexpr (has_exception_hook<R>)
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

// The left arm of `|` provably starves the right when both are catch_only wrappers, the left's
// guard list claims every type the right lists, and the left's inner policy never declines a
// claimed exception. Sound and incomplete, like every dead-code theorem here: nested
// composites and raw guards escape the pattern; an inner policy that can decline (nothrow
// test fails) keeps the right arm live.
template <class L, class R>
inline constexpr bool right_arm_starved = false;

template <class P1, class... As, class P2, class... Bs>
inline constexpr bool right_arm_starved<catch_only_t<P1, As...>, catch_only_t<P2, Bs...>> =
  exception_path_nothrow_v<P1> && (claimed_by_any<Bs, As...> && ...);

// A cone on the left starves an exact entry inside it on the right; an exact entry on the
// left starves only its own repetitions. The converse (exact left, cone right) never starves:
// the cone always has more members.
template <class P1, class... As, class P2, class... Bs>
inline constexpr bool right_arm_starved<catch_only_t<P1, As...>, catch_exactly_t<P2, Bs...>> =
  exception_path_nothrow_v<P1> && (claimed_by_any<Bs, As...> && ...);

template <class P1, class... As, class P2, class... Bs>
inline constexpr bool right_arm_starved<catch_exactly_t<P1, As...>, catch_exactly_t<P2, Bs...>> =
  exception_path_nothrow_v<P1> && (listed_exactly<Bs, As...> && ...);

// The alternation composite `L | R`: `L` claims first; if it declines by throwing, `R`
// handles the original (re-observed) exception. Each arm is called at the uniform 3-arg shape;
// acceptance is interpreted at `decltype(fn())`.
template <class L, class R>
struct policy_or : composite_hooks<L, R>
{
  using exception_sink_tag = void;

  static_assert(has_exception_hook<L> && has_exception_hook<R>,
                "both sides of | must answer the exception path (have an exception hook)");
  static_assert(!exception_path_nothrow_v<L>,
                "the left policy never declines; alternatives after it are unreachable");
  static_assert(!right_arm_starved<L, R>,
                "the left type guard already claims every exception type the right arm lists; "
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

    // Interpret both arms at the callable's result type. Void callables surface ignore so
    // this composite can still sit as a top-level policy.
    CCCL_TRY
    {
      if constexpr (::cuda::std::is_void_v<Raw>)
      {
        interpret_answer<Raw>(this->l_, exception, loc, fn);
        return ::std::ignore;
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
        return ::std::ignore;
      }
      else
      {
        return reobserve_right();
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
      return ::std::ignore;
    }
    else
    {
      return go(go, exception, n_);
    }
    CCCL_DIAG_POP
  }
};

// --- The conversion law (SPEC-ADDENDUM-7 commit 4) -----------------------------------------
//
// One law at three horizons: value-preserving conversions only. Enforced here at compile time
// for concrete policies; the erased sink enforces the same law by type range at first use and
// by the actual value at first throw. Integrals convert when the destination's range contains
// the source's; anything arithmetic converts to floating (precision loss is tolerated where
// range loss is not); floating never converts to integral; non-arithmetic pairs follow the
// ordinary implicit-conversion rules. When narrowing is intended, write the conversion in the
// policy -- subst(0xffffffffu), not subst(-1) -- so the intent is visible at the callsite.

template <class T, bool = ::cuda::std::is_enum_v<T>>
struct integral_base
{
  using type = T;
};
template <class T>
struct integral_base<T, true>
{
  using type = ::cuda::std::underlying_type_t<T>;
};

template <class From, class To>
constexpr bool value_preserving_impl()
{
  using F = typename integral_base<::cuda::std::remove_cvref_t<From>>::type;
  using T = ::cuda::std::remove_cvref_t<To>;
  if constexpr (!::cuda::std::is_arithmetic_v<F> || !::cuda::std::is_arithmetic_v<T>)
  {
    return true; // non-arithmetic pairs: the is_convertible baseline is the whole law
  }
  else if constexpr (::cuda::std::is_floating_point_v<T>)
  {
    return true; // precision loss is tolerated where range loss is not
  }
  else if constexpr (::cuda::std::is_floating_point_v<F>)
  {
    return false; // floating never converts to integral
  }
  else
  {
    // Integral range containment, sign-aware. A signed source holds negatives an unsigned
    // destination cannot; equal signedness compares widths; unsigned-to-signed needs strictly
    // more width to cover the source's maximum.
    constexpr bool f_signed = ::cuda::std::is_signed_v<F>;
    constexpr bool t_signed = ::cuda::std::is_signed_v<T>;
    if constexpr (f_signed && !t_signed)
    {
      return false;
    }
    else if constexpr (f_signed == t_signed)
    {
      return sizeof(F) <= sizeof(T);
    }
    else
    {
      return sizeof(F) < sizeof(T);
    }
  }
}

template <class From, class To>
inline constexpr bool value_preserving_v =
  ::cuda::std::is_convertible_v<From, To> && value_preserving_impl<From, To>();

// Interpret the final element's answer as the expression's value, converting to `Expr`.
template <class Expr, class P, class Fn>
Expr interpret_answer(
  P& policy, const ::std::exception* exception, const ::cuda::std::source_location loc, Fn& fn)
{
  using Answer = hook_answer_t<P, Fn>;
  static_assert(!::cuda::std::is_void_v<Answer>,
                "the final policy must answer the exception path: nullval to die, ::std::ignore "
                "to resume, or a value to substitute");

  if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<Answer>, nullval>)
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
                  "nullval, like abort and terminate), ::std::ignore, or a value convertible to "
                  "the result of the callable");
    static_assert(value_preserving_v<Answer, Expr>,
                  "the policy's answer does not preserve the callable's value range (for example "
                  "an int answer under an unsigned result); write the conversion in the policy -- "
                  "subst(0xffffffffu), not subst(-1) -- if the narrowing is intended");
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
  Reaction reaction_;
  const ::cuda::std::source_location loc_;
};

template <class R>
on_throw_policy(R, ::cuda::std::source_location) -> on_throw_policy<R>;

template <class Reaction, class Fn>
// A resuming chain reads neither exception nor location in some instantiations; gcc 9 flags the
// unread policy without the attribute.
decltype(auto) operator<<([[maybe_unused]] on_throw_policy<Reaction> policy,
                          Fn&& fn) noexcept(exception_path_nothrow_v<Reaction, Fn>
                                               && on_enter_nothrow_v<Reaction>)
{
  // Bind as a non-const lvalue: a hook may invoke it again later.
  Fn& f = fn;

  // A `noexcept` callable puts the policy out of reach: an exception raised inside it ends the
  // program where it stands, so the catch below could never run and the policy would be a
  // promise nobody keeps.
  static_assert(!noexcept(f()),
                "on_throw has nothing to do for a noexcept callable, which terminates rather than "
                "throws; call such a callable directly");

  using Expr = decltype(f());
  using P    = Reaction;

  // The entry gate runs before the attempt and outside the policy's own catch: an exception
  // thrown here (a gate refusing the attempt) belongs to the enclosing scope, never to the
  // policy that raised it.
  if constexpr (has_on_enter<P>)
  {
    static_assert(::cuda::std::is_void_v<on_enter_of<P>>,
                  "on_enter answers nothing: it admits the attempt or refuses it by throwing");
    policy.reaction_.on_enter();
  }

  if constexpr (::cuda::std::is_void_v<Expr>)
  {
    CCCL_TRY
    {
      f();
      if constexpr (has_on_success_void<P>)
      {
        using Answer = decltype(::cuda::std::declval<P&>().on_success());
        static_assert(::cuda::std::is_same_v<Answer, Expr>,
                      "a policy's on_success must preserve the expression type; policies no "
                      "longer own it (SPEC-ADDENDUM-7)");
        policy.reaction_.on_success();
      }
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
    CCCL_TRY
    {
      if constexpr (has_on_success_with<P, Expr>)
      {
        using Answer = decltype(::cuda::std::declval<P&>().on_success(::cuda::std::declval<Expr>()));
        static_assert(::cuda::std::is_same_v<Answer, Expr>,
                      "a policy's on_success must preserve the expression type; policies no "
                      "longer own it (SPEC-ADDENDUM-7)");
        return policy.reaction_.on_success(f());
      }
      else
      {
        return f();
      }
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
 * @brief Restricts a policy to exceptions whose dynamic type is exactly one of `E1, E2, ...`:
 * monomorphic where @ref catch_only is polymorphic, so a handler accepts a type without
 * inheriting its cone. `catch_exactly<std::bad_alloc>(p)` handles allocation pressure yet
 * lets `std::bad_array_new_length`, a size-computation bug, fly on; value operations that
 * would slice under a cone (copy, store) are safe behind an exact gate; and the guard's
 * contract cannot drift when someone derives a new type later. Matching reads the dynamic
 * type through the `std::exception` funnel, so every listed type must derive
 * `std::exception`, and a non-std active exception always declines. Duplicates are rejected;
 * Base and Derived may both be listed, each matching only itself. In `|` chains,
 * `catch_exactly<E>(recover) | catch_only<E>(fallback)` layers the exact type against the
 * rest of its cone; the reverse order starves the exact arm and is a compile error.
 */
template <class... Es, class P>
auto catch_exactly(P&& p)
{
  static_assert(sizeof...(Es) > 0, "catch_exactly requires at least one exception type");
  static_assert((::cuda::std::is_base_of_v<::std::exception, Es> && ...),
                "catch_exactly matches dynamic types through the std::exception funnel; every "
                "listed type must derive std::exception (catch_only takes anything catchable)");
  static_assert(detail::catch_exactly_pack_ok<Es...>,
                "catch_exactly<..., E, ..., E, ...>: a repeated entry is dead");
  auto np = detail::normalize(::cuda::std::forward<P>(p));
  return detail::catch_exactly_t<decltype(np), Es...>{::cuda::std::move(np)};
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
 * Returns `when_t{pred} & p`: as a `|` arm, false means the next alternative gets the
 * exception; inside a larger `&`, false declines the whole sequence.
 */
template <class Pred, class P>
auto when(Pred&& pred, P&& p)
{
  return when_t<Pred>{::cuda::std::forward<Pred>(pred)} & detail::normalize(::cuda::std::forward<P>(p));
}

/**
 * @brief Runs the head, then every finalizer, on the exception path -- the finalizers run
 * whether the head accepts or declines. The composite's answer is the HEAD's; finalizers'
 * answers are discarded. If the head declines, the finalizers observe the in-flight
 * exception and the head's exception then continues onward (a finalizer's own throw
 * replaces it, as anywhere in C++). Later arguments are increasingly unconditional.
 * A named function, not an operator: it has no left identity.
 */
template <class A, class B>
struct always_t
{
  //! @cond
  using exception_sink_tag = void;
  //! @endcond
  A head_;
  B fin_;

  template <class Fn>
  decltype(auto) operator()(const ::std::exception* e, const ::cuda::std::source_location loc, Fn& fn)
  {
    CCCL_TRY
    {
      using Ans = detail::hook_answer_t<A, Fn>;
      if constexpr (::cuda::std::is_void_v<Ans>)
      {
        head_(e, loc, fn);
        fin_(e, loc, fn);
        return;
      }
      else if constexpr (detail::answers_nothing<A, Fn>)
      {
        // Never actually returns (the decline path below runs the finalizer); returning the
        // call keeps the composite's answer type `nullval` via guaranteed elision.
        return head_(e, loc, fn);
      }
      else
      {
        decltype(auto) r = head_(e, loc, fn);
        static_cast<void>(fin_(e, loc, fn)); // a finalizer's answer is discarded
        return static_cast<Ans>(r);
      }
    }
    CCCL_CATCH_ALL
    {
      // The head declined (or a finalizer threw after an accept): run the finalizer with the
      // CURRENT exception, then let that exception continue onward.
      CCCL_TRY
      {
        throw;
      }
      CCCL_CATCH (const ::std::exception& cur)
      {
        static_cast<void>(fin_(&cur, loc, fn));
      }
      CCCL_CATCH_ALL
      {
        static_cast<void>(fin_(nullptr, loc, fn));
      }
      throw;
    }
  }

  // The head's success hook is a type-preserving side-effect; forward it.
  template <class R, ::cuda::std::enable_if_t<detail::has_on_success_with<A, R>, int> = 0>
  decltype(auto) on_success(R&& r)
  {
    return head_.on_success(::cuda::std::forward<R>(r));
  }

  template <class Self = A, ::cuda::std::enable_if_t<detail::has_on_success_void<Self>, int> = 0>
  decltype(auto) on_success()
  {
    return head_.on_success();
  }

  // Entry gates forward to head and finalizer alike: each may refuse the attempt.
  template <class AA                                                                               = A,
            class BB                                                                               = B,
            ::cuda::std::enable_if_t<detail::has_on_enter<AA> || detail::has_on_enter<BB>, int> = 0>
  void on_enter() noexcept(detail::on_enter_nothrow_v<A> && detail::on_enter_nothrow_v<B>)
  {
    if constexpr (detail::has_on_enter<A>)
    {
      head_.on_enter();
    }
    if constexpr (detail::has_on_enter<B>)
    {
      fin_.on_enter();
    }
  }
};

//! @brief See @ref always_t. Variadic: `always(a, b, c)` folds left, so both `b` and `c` run
//! regardless of `a`, and `c` runs regardless of `b`.
template <class A, class B, class... Cs>
auto always(A&& a, B&& b, Cs&&... cs)
{
  if constexpr (sizeof...(Cs) == 0)
  {
    auto ha = detail::normalize(::cuda::std::forward<A>(a));
    auto hb = detail::normalize(::cuda::std::forward<B>(b));
    return always_t<decltype(ha), decltype(hb)>{::cuda::std::move(ha), ::cuda::std::move(hb)};
  }
  else
  {
    return always(always(::cuda::std::forward<A>(a), ::cuda::std::forward<B>(b)),
                  ::cuda::std::forward<Cs>(cs)...);
  }
}
/**
 * @brief A type-erased exception policy: any policy behind one concrete type.
 *
 * The shell interprets at `decltype(fn())`. Erasure boxes answers as `std::any`
 * internally; that box reaches the user only when the callable itself returns
 * `std::any`. Retrying policies keep their internal loops, state, and
 * rethrow-on-exhaustion; composites erase whole; a sink composes with `&`, `|`,
 * `*`, `when`, `always` and re-erases.
 *
 * Custom runtime policies derive from the public @ref sink_base and implement
 * `hook` (and `clone`); `on_success` defaults to identity. Hand-written models
 * default to `passthrough` and are unchecked. Erased |-composites are
 * `passthrough` (a loud, value-checked unbox on first throw is their
 * backstop); a pure-& composite keeps its final policy's precise kind and
 * is checked at first use. A wrapped
 * success channel that cannot accept `std::any` (for example `remember`) does
 * not survive erasure.
 *
 * LIMIT: `std::any` cannot carry references. A reference-returning callable is
 * rejected at the composition site; use a concrete policy, or return a pointer.
 */
class exception_sink
{
public:
  //! @brief What the wrapped policy's answer was, statically, at erasure time.
  enum class answer_kind
  {
    dies, //!< the hook never returns (answered `nullval`)
    resumes, //!< resume (answered `std::ignore`)
    effects, //!< side effect only (answered `void`)
    integral, //!< a stored integral or enumeration, with a value range
    floating, //!< a stored floating value (precision loss under a floating body is tolerated)
    udt, //!< a stored class, pointer, or other exact-match type
    passthrough //!< the boxed value is the callable's own result (`std::any`)
  };

  //! @brief The erasure surface. Public: custom runtime policies derive from
  //! it and implement `hook` (rethrow by throwing; hand back `std::any`,
  //! empty meaning "no value") and `clone`; `on_success` defaults to identity.
  struct sink_base
  {
    const answer_kind kind;
    //! Whether the exception path may rethrow (`false`: alternatives after
    //! this sink are unreachable). Conservative for hand-written models.
    const bool may_rethrow;
    //! Inclusive range of a stored integral answer; unused for other kinds.
    const long long min_value;
    const unsigned long long max_value;
    //! Exact stored type for `udt` checks; `nullptr` encodes `passthrough`.
    const ::std::type_info* answer_type;
    const ::std::string_view answer_name;

    sink_base(answer_kind kind                    = answer_kind::passthrough,
              bool may_rethrow                    = true,
              long long min_value                 = 0,
              unsigned long long max_value        = 0,
              const ::std::type_info* answer_type = nullptr,
              ::std::string_view answer_name      = {})
        : kind(kind)
        , may_rethrow(may_rethrow)
        , min_value(min_value)
        , max_value(max_value)
        , answer_type(answer_type)
        , answer_name(answer_name)
    {}
    virtual ~sink_base()             = default;
    virtual sink_base* clone() const = 0;
    virtual ::std::any hook(const ::std::exception*, ::cuda::std::source_location, ::std::function<::std::any()>&) = 0;
    virtual ::std::any on_success(::std::any boxed)
    {
      return boxed;
    }
    virtual ::std::any on_success()
    {
      return {};
    }

    //! @brief Entry gate, run before each attempt. Default: admit.
    virtual void on_enter() {}
  };

private:
  template <class Ans>
  static constexpr answer_kind kind_of()
  {
    using T = ::cuda::std::remove_cvref_t<Ans>;
    if constexpr (::cuda::std::is_void_v<T>)
    {
      return answer_kind::effects;
    }
    else if constexpr (detail::is_ignore_v<T>)
    {
      return answer_kind::resumes;
    }
    else if constexpr (::cuda::std::is_same_v<T, nullval>)
    {
      return answer_kind::dies;
    }
    else if constexpr (::cuda::std::is_same_v<T, ::std::any>)
    {
      return answer_kind::passthrough;
    }
    else if constexpr (::cuda::std::is_enum_v<T> || ::cuda::std::is_integral_v<T>)
    {
      return answer_kind::integral;
    }
    else if constexpr (::cuda::std::is_floating_point_v<T>)
    {
      return answer_kind::floating;
    }
    else
    {
      return answer_kind::udt;
    }
  }

  template <class Int>
  static constexpr long long limits_min()
  {
    if constexpr (::cuda::std::is_unsigned_v<Int>)
    {
      return 0;
    }
    else
    {
      return static_cast<long long>(::std::numeric_limits<Int>::min());
    }
  }

  template <class Int>
  static constexpr unsigned long long limits_max()
  {
    return static_cast<unsigned long long>(::std::numeric_limits<Int>::max());
  }

  template <class Ans>
  static constexpr long long stored_min()
  {
    using T = ::cuda::std::remove_cvref_t<Ans>;
    if constexpr (kind_of<T>() == answer_kind::integral)
    {
      if constexpr (::cuda::std::is_enum_v<T>)
      {
        return limits_min<::cuda::std::underlying_type_t<T>>();
      }
      else
      {
        return limits_min<T>();
      }
    }
    else
    {
      return 0;
    }
  }

  template <class Ans>
  static constexpr unsigned long long stored_max()
  {
    using T = ::cuda::std::remove_cvref_t<Ans>;
    if constexpr (kind_of<T>() == answer_kind::integral)
    {
      if constexpr (::cuda::std::is_enum_v<T>)
      {
        return limits_max<::cuda::std::underlying_type_t<T>>();
      }
      else
      {
        return limits_max<T>();
      }
    }
    else
    {
      return 0;
    }
  }

  //! The one universal erasure path: instantiate the wrapped policy's
  //! templated hook at the boxing proxy; derive all metadata statically.
  template <class P>
  struct model final : sink_base
  {
    P p_;

    using ans_t                        = detail::hook_answer_t<P, ::std::function<::std::any()>>;
    static constexpr answer_kind skind = kind_of<ans_t>();

    explicit model(P p)
        : sink_base(
            skind,
            !detail::exception_path_nothrow_v<P, ::std::function<::std::any()>>,
            stored_min<ans_t>(),
            stored_max<ans_t>(),
            (skind == answer_kind::passthrough || skind == answer_kind::dies || skind == answer_kind::resumes
             || skind == answer_kind::effects)
              ? nullptr
              : &typeid(::cuda::std::remove_cvref_t<ans_t>),
            type_name<::cuda::std::remove_cvref_t<ans_t>>)
        , p_(::cuda::std::move(p))
    {}

    sink_base* clone() const override
    {
      return new model(*this); // a fresh policy copy: re-armed state
    }

    ::std::any
    hook(const ::std::exception* e, ::cuda::std::source_location loc, ::std::function<::std::any()>& fn) override
    {
      if constexpr (skind == answer_kind::effects || skind == answer_kind::resumes)
      {
        static_cast<void>(p_(e, loc, fn));
        return {};
      }
      else if constexpr (skind == answer_kind::dies)
      {
        static_cast<void>(p_(e, loc, fn));
        CCCL_UNREACHABLE();
      }
      else if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<ans_t>, ::std::any>)
      {
        return p_(e, loc, fn); // the value flowed through the callable; no re-box
      }
      else
      {
        return ::std::any(p_(e, loc, fn)); // a stored value, boxed as its own type
      }
    }

    ::std::any on_success(::std::any boxed) override
    {
      if constexpr (detail::has_on_success_with<P, ::std::any>)
      {
        return ::std::any(p_.on_success(::cuda::std::move(boxed)));
      }
      else
      {
        // Identity. NOTE: a wrapped success channel that cannot accept
        // `std::any` (it needs the real type, like remember's store) lands
        // here too -- its success feature does not survive erasure.
        return boxed;
      }
    }
    ::std::any on_success() override
    {
      if constexpr (detail::has_on_success_void<P>)
      {
        if constexpr (::cuda::std::is_void_v<decltype(p_.on_success())>)
        {
          p_.on_success();
          return {};
        }
        else
        {
          return ::std::any(p_.on_success());
        }
      }
      else
      {
        return {};
      }
    }
    void on_enter() override
    {
      if constexpr (detail::has_on_enter<P>)
      {
        p_.on_enter();
      }
    }
  };

  ::std::unique_ptr<sink_base> p_;

  [[noreturn]] void throw_mismatch(::std::string_view stored, ::std::string_view wanted) const
  {
    ::std::string msg{"exception_sink cannot convert "};
    msg.append(stored.data(), stored.size());
    msg.append(" answer to ");
    msg.append(wanted.data(), wanted.size());
    throw ::std::logic_error(msg);
  }

  template <class Int>
  [[nodiscard]] bool range_contains_int() const
  {
    const auto raw_max = ::std::numeric_limits<Int>::max();
    if constexpr (::cuda::std::is_unsigned_v<Int>)
    {
      if (p_->min_value < 0)
      {
        return false;
      }
      return p_->max_value <= static_cast<unsigned long long>(raw_max);
    }
    else
    {
      const auto raw_min = ::std::numeric_limits<Int>::min();
      if (p_->min_value < static_cast<long long>(raw_min))
      {
        return false;
      }
      return p_->max_value <= static_cast<unsigned long long>(raw_max);
    }
  }

  template <class Raw>
  [[nodiscard]] bool range_contains() const
  {
    using T = ::cuda::std::remove_cvref_t<Raw>;
    if constexpr (::cuda::std::is_enum_v<T>)
    {
      return range_contains_int<::cuda::std::underlying_type_t<T>>();
    }
    else
    {
      return range_contains_int<T>();
    }
  }

  template <class Raw>
  void check_compatible() const
  {
    using T                             = ::cuda::std::remove_cvref_t<Raw>;
    const answer_kind kind             = p_->kind;
    [[maybe_unused]] const auto wanted = type_name<T>;
    [[maybe_unused]] const auto stored =
      p_->answer_name.empty() ? ::std::string_view{"<erased>"} : p_->answer_name;
    if (kind == answer_kind::dies || kind == answer_kind::resumes || kind == answer_kind::effects
        || kind == answer_kind::passthrough)
    {
      return;
    }
    if constexpr (::cuda::std::is_void_v<Raw>)
    {
      throw_mismatch(stored, wanted);
    }
    else if (kind == answer_kind::integral)
    {
      if constexpr (::cuda::std::is_floating_point_v<T>)
      {
        return;
      }
      else if constexpr (::cuda::std::is_integral_v<T> || ::cuda::std::is_enum_v<T>)
      {
        if (range_contains<Raw>())
        {
          return;
        }
      }
      throw_mismatch(stored, wanted);
    }
    else if (kind == answer_kind::floating)
    {
      if constexpr (::cuda::std::is_floating_point_v<T>)
      {
        return;
      }
      throw_mismatch(stored, wanted);
    }
    else if (kind == answer_kind::udt)
    {
      if (p_->answer_type && *p_->answer_type == typeid(T))
      {
        return;
      }
      throw_mismatch(stored, wanted);
    }
  }

  template <class Raw>
  [[nodiscard]] Raw unbox(const ::std::any& box) const
  {
    using T = ::cuda::std::remove_cvref_t<Raw>;
    if constexpr (::cuda::std::is_same_v<T, ::std::any>)
    {
      return box;
    }
    else if (!box.has_value())
    {
      if constexpr (::cuda::std::is_default_constructible_v<T>)
      {
        return T{};
      }
      else
      {
        throw_mismatch("<empty>", type_name<T>);
      }
    }
    else if (const T* exact = ::std::any_cast<T>(&box))
    {
      return *exact;
    }
    else if constexpr (::cuda::std::is_arithmetic_v<T> || ::cuda::std::is_enum_v<T>)
    {
      T out{};
      bool hit         = false;
      bool found_lossy = false;
      // The same conversion law, per VALUE: the box is opaque to the first-use type check
      // (passthrough composites), so representability is decided on the number itself.
      const auto fits = [](auto v) -> bool {
        using B = typename detail::integral_base<T>::type;
        if constexpr (::cuda::std::is_signed_v<decltype(v)>)
        {
          if (v < 0)
          {
            if constexpr (::cuda::std::is_signed_v<B>)
            {
              return static_cast<long long>(v) >= static_cast<long long>(::std::numeric_limits<B>::min());
            }
            else
            {
              return false;
            }
          }
        }
        return static_cast<unsigned long long>(v)
            <= static_cast<unsigned long long>(::std::numeric_limits<B>::max());
      };
      const auto accept = [&](auto tag) {
        using Stored = typename decltype(tag)::type;
        if (hit || found_lossy)
        {
          return;
        }
        if (const Stored* p = ::std::any_cast<Stored>(&box))
        {
          if constexpr (::cuda::std::is_floating_point_v<typename detail::integral_base<T>::type>)
          {
            out = static_cast<T>(*p); // anything -> floating: by fiat
            hit = true;
          }
          else if constexpr (::cuda::std::is_floating_point_v<Stored>)
          {
            found_lossy = true; // floating never converts to integral
          }
          else if (fits(*p))
          {
            out = static_cast<T>(*p);
            hit = true;
          }
          else
          {
            found_lossy = true; // right category, unrepresentable value
          }
        }
      };
      accept(::cuda::std::type_identity<bool>{});
      accept(::cuda::std::type_identity<char>{});
      accept(::cuda::std::type_identity<signed char>{});
      accept(::cuda::std::type_identity<unsigned char>{});
      accept(::cuda::std::type_identity<short>{});
      accept(::cuda::std::type_identity<unsigned short>{});
      accept(::cuda::std::type_identity<int>{});
      accept(::cuda::std::type_identity<unsigned>{});
      accept(::cuda::std::type_identity<long>{});
      accept(::cuda::std::type_identity<unsigned long>{});
      accept(::cuda::std::type_identity<long long>{});
      accept(::cuda::std::type_identity<unsigned long long>{});
      accept(::cuda::std::type_identity<float>{});
      accept(::cuda::std::type_identity<double>{});
      accept(::cuda::std::type_identity<long double>{});
      if (!hit)
      {
        throw_mismatch(box.type().name(), type_name<T>);
      }
      return out;
    }
    else
    {
      throw_mismatch(box.type().name(), type_name<T>);
    }
  }

public:
  //! @cond
  using exception_sink_tag = void;
  //! @endcond

  //! @brief Erase a policy. Prefer the @ref type_erase factory, which also
  //! normalizes the historical reactions.
  template <class P,
            ::cuda::std::enable_if_t<detail::is_exception_sink_v<P>
                                       && !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<P>, exception_sink>,
                                     int> = 0>
  explicit exception_sink(P p)
      : p_(new model<P>(::cuda::std::move(p)))
  {}

  //! @brief Adopt a custom model derived from @ref sink_base.
  explicit exception_sink(::std::unique_ptr<sink_base> custom)
      : p_(::std::move(custom))
  {
    CCCL_ASSERT(p_, "exception_sink requires a non-null model");
  }

  exception_sink(const exception_sink& other)
      : p_(other.p_->clone())
  {}
  exception_sink(exception_sink&&) noexcept            = default;
  exception_sink& operator=(exception_sink&&) noexcept = default;
  exception_sink& operator=(const exception_sink& other)
  {
    p_.reset(other.p_->clone());
    return *this;
  }

  [[nodiscard]] answer_kind kind() const noexcept
  {
    return p_->kind;
  }
  [[nodiscard]] bool may_rethrow() const noexcept
  {
    return p_->may_rethrow;
  }

  //! @brief The uniform hook: answers `decltype(fn())`. A void callable answers
  //! `decltype(::std::ignore)` so the presence-probe archetype stays admissible.
  template <class Fn, class Raw = decltype(::cuda::std::declval<Fn&>()())>
  auto operator()(const ::std::exception* e, const ::cuda::std::source_location loc, Fn& fn)
    -> ::cuda::std::conditional_t<::cuda::std::is_void_v<Raw>, decltype(::std::ignore), Raw>
  {
    static_assert(!::cuda::std::is_reference_v<Raw>,
                  "exception_sink cannot serve a reference-returning callable: std::any cannot "
                  "carry references; use a concrete policy, or return a pointer");
    if constexpr (::cuda::std::is_reference_v<Raw>)
    {
      CCCL_UNREACHABLE();
    }
    else
    {
      ::std::function<::std::any()> proxy = [&fn]() -> ::std::any {
        if constexpr (::cuda::std::is_void_v<Raw>)
        {
          fn();
          return {};
        }
        else
        {
          return ::std::any(fn());
        }
      };

      if constexpr (::cuda::std::is_void_v<Raw>)
      {
        check_compatible<Raw>();
        static_cast<void>(p_->hook(e, loc, proxy));
        return ::std::ignore;
      }
      else if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<Raw>, ::std::any>)
      {
        return p_->hook(e, loc, proxy);
      }
      else
      {
        check_compatible<Raw>();
        return unbox<Raw>(p_->hook(e, loc, proxy));
      }
    }
  }

  //! @brief Entry gate: forwards to the erased policy (no-op when it has none).
  void on_enter()
  {
    p_->on_enter();
  }

  //! @brief Type-preserving success passthrough: box, delegate, unbox to the same type.
  template <class R>
  R on_success(R&& r)
  {
    static_assert(!::cuda::std::is_reference_v<R>,
                  "exception_sink cannot serve a reference-returning callable: std::any cannot "
                  "carry references; use a concrete policy, or return a pointer");
    if constexpr (::cuda::std::is_reference_v<R>)
    {
      CCCL_UNREACHABLE();
    }
    else
    {
      check_compatible<R>();
      return unbox<R>(p_->on_success(::std::any(::cuda::std::forward<R>(r))));
    }
  }
  void on_success()
  {
    static_cast<void>(p_->on_success());
  }
};

//! @brief Erase any policy (or historical reaction) into an @ref exception_sink.
template <class P>
exception_sink type_erase(P&& p)
{
  return exception_sink{detail::normalize(::cuda::std::forward<P>(p))};
}
} // namespace exception_policies

// The abort tripwire that once lived at this spot is retired. It caught bare abort() calls
// during the migration of the policy vocabulary into exception_policies, when such calls
// could silently rebind; with the policies in a non-inline namespace, a bare abort() in
// this scope can only mean ::abort, which is what callers expect. Policy uses spell
// exception_policies::abort (or arrive through ON_THROW, which injects the namespace for
// the policy expression only).

/**
 * @brief Creates a policy saying how to react if a callable throws.
 *
 * Apply the policy with `on_throw(policy) << callable`. Its expression type is always
 * `decltype(callable())`; no policy changes it.
 *
 * A policy is an object exposing any of two optional capabilities, discovered by compile-time
 * introspection: the exception hook
 * `(const std::exception*, source_location, Fn&)` whose return value is its answer on the throw
 * path (the callable may be re-invoked by policies like `retry`; most policies ignore it), and
 * a success hook `on_success(...)` that observes the result while preserving its type. The named policies
 * include @ref exception_policies::notify_t "notify", @ref exception_policies::subst_t "subst", @ref
 * exception_policies::defer_t "defer",
 * @ref exception_policies::rethrow_t "rethrow", @ref exception_policies::retry_t "retry", @ref
 * exception_policies::as_expected_t "as_expected", @ref exception_policies::noop_t "noop", @ref
 * exception_policies::catch_only, @ref exception_policies::catch_exactly,
 * @ref exception_policies::when "when",
 * @ref exception_policies::translate_t "translate" / @ref exception_policies::nest, @ref exception_policies::delay_t
 * "delay", @ref exception_policies::backoff, and
 * @ref exception_policies::remember_t "remember", @ref exception_policies::circuit_breaker_t
 * "circuit_breaker", and @ref exception_policies::always. Guards decline what they do not
 * claim; translators decline with a different exception; delay/backoff/retry re-run; remember serves the last success.
 * Policies compose with `&` (sequence; the last element answers; non-final answers are
 * discarded) and `|` (alternation; the left may decline by throwing), and with `*` (n-fold
 * `|`).
 *
 * For backward compatibility `on_throw` also accepts non-policy reactions: `std::ignore`
 * resumes with a default-constructed result; and anything else is taken as a substitution
 * value, exactly as `subst(value)` (including a user's nullary `nullval`-returning ending,
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
 * configures sinks, or a namespace alias (`namespace pol = cuda::experimental::stf::exception_policies;`):
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
#  error "CUDASTF's exception_policy.cuh defines ON_THROW; rename the prior definition"
#endif
//! @brief Statement-shaped on_throw: ON_THROW(policy-expression) { body };
//! The policy expression is evaluated with `exception_policies` visible, so
//! ON_THROW(notify & retry * 3 | subst(-1)) { return flaky(); }; needs no
//! qualification. All arguments forward to on_throw, so a source_location
//! may follow the policy: ON_THROW(notify, loc) { body };. Expands to
//! on_throw(...) << a reference-capturing lambda; the call-site location is
//! captured exactly as with plain on_throw. The macro ends at `[&]()`:
//! supply the body type by composition when needed, as in
//! ON_THROW(retry | subst(-1)) -> int { throw failure(); };.
#define ON_THROW(...)                                              \
  [&] {                                                            \
    using namespace ::cuda::experimental::stf::exception_policies; \
    return ::cuda::experimental::stf::on_throw(VA_ARGS__);       \
  }() << [&]()

#ifdef UNITTESTED_FILE
UNITTEST("nullval")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
  // No values: not constructible in any way.
  static_assert(!::std::is_default_constructible_v<nullval>);
  static_assert(!::std::is_copy_constructible_v<nullval>);
  static_assert(!::std::is_move_constructible_v<nullval>);
  // One-way conversions: `nullval` converts to every type, no type converts to `nullval`.
  static_assert(::std::is_convertible_v<nullval, int>);
  static_assert(::std::is_convertible_v<nullval, int&>);
  static_assert(::std::is_convertible_v<nullval, void (*)()>);
  static_assert(!::std::is_convertible_v<int, nullval>);
  // A never-returning call may be returned from a function of any result type, references
  // included; the conversion typechecks and never runs.
  const auto never = []() -> nullval {
    ::std::abort();
  };
  [[maybe_unused]] const auto propagates = [&]() -> int& {
    return never();
  };
  // A `nullval` expression also supplies one arm of a ternary, the other arm setting the type.
  const auto pick = [&](bool ok) -> int {
    return ok ? 42 : never();
  };
  EXPECT(pick(true) == 42);
};

UNITTEST("circuit_breaker")
{
  using namespace cuda::experimental::stf;
  namespace pol = cuda::experimental::stf::exception_policies;

  auto budget = ::std::make_shared<int>(2);
  // A NAMED policy value: the entry gate still fires per attempt, not per construction.
  auto guarded = pol::circuit_breaker(budget) & pol::subst(-1);

  int runs   = 0;
  auto flaky = [&]() -> int {
    ++runs;
    throw ::std::runtime_error("down");
  };

  // Two failures spend the budget; each answers through subst.
  EXPECT((on_throw(guarded) << flaky) == -1);
  EXPECT((on_throw(guarded) << flaky) == -1);
  EXPECT(*budget == 0);

  // Third attempt: refused at the gate, the body never runs, circuit_open escapes.
  bool gated = false;
  CCCL_TRY
  {
    on_throw(guarded) << flaky;
  }
  CCCL_CATCH ([[maybe_unused]] const pol::circuit_open& open)
  {
    gated = true;
  }
  CCCL_CATCH_ALL
  {
    EXPECT(false, "the gate must refuse with circuit_open, nothing else");
  }
  EXPECT(gated);
  EXPECT(runs == 2);

  // External administration: refill through the shared int, then a success restores the
  // budget to its creation-time value.
  *budget = 1;
  EXPECT((on_throw(guarded) << [] () -> int { return 7; }) == 7);
  EXPECT(*budget == 2);

  // The macro spelling gates identically.
  *budget = 0;
  gated = false;
  CCCL_TRY
  {
    ON_THROW(circuit_breaker(budget) & subst(-1)) {
      return 9;
    };
  }
  CCCL_CATCH ([[maybe_unused]] const pol::circuit_open& open)
  {
    gated = true;
  }
  CCCL_CATCH_ALL
  {
    EXPECT(false, "the gate must refuse with circuit_open, nothing else");
  }
  EXPECT(gated);

  // The erased form carries the gate through: sinks re-erase, gates survive.
  *budget = 0;
  pol::exception_sink erased = pol::type_erase(pol::circuit_breaker(budget) & pol::subst(-1));
  gated                 = false;
  CCCL_TRY
  {
    on_throw(erased) << flaky;
  }
  CCCL_CATCH ([[maybe_unused]] const pol::circuit_open& open)
  {
    gated = true;
  }
  CCCL_CATCH_ALL
  {
    EXPECT(false, "the erased gate must refuse with circuit_open, nothing else");
  }
  EXPECT(gated);
  EXPECT(runs == 2);

  // A gate that can throw removes noexcept from the whole expression.
  static_assert(!noexcept(on_throw(guarded) << flaky));
};

// Negative-compile expectations (do not compile; kept as comments near the code they guard):
//  - exception_policies::abort();                  // the policy has no bare-call form; hooks only
//  - on_throw(abort & notify) << [] {};            // "policies after a never-returning policy are unreachable"
//  - on_throw(notify) << []() noexcept {};         // existing rule, unchanged message
//  - on_throw(notify & subst(42)) << []() -> int& {...}; // reference result vs owned substitution (existing rule)
//  - on_throw(as_expected) << []() -> int { return 1; };
//      // "as_expected requires the callable to return a cuda::std::expected instantiation"
//  - a policy whose on_success returns a different type than decltype(fn());
//      // "a policy's on_success must preserve the expression type; policies no longer own it (SPEC-ADDENDUM-7)"
//  - on_throw(type_erase(subst(1))) << []() -> int& { static int x = 0; return x; };
//      // "exception_sink cannot serve a reference-returning callable: std::any cannot carry references; use a concrete
//      policy, or return a pointer"
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

  // A terminating handler declares `nullval` and dies on its own terms; it stays out of the
  // way as long as nothing throws. Raw lambdas of the right shape are policies, no wrapping.
  const auto die = [](const ::std::exception*, ::cuda::std::source_location, auto&) noexcept -> nullval {
    ::std::abort();
  };
  const int untouched = on_throw(die) << [] {
    return 7;
  };
  EXPECT(untouched == 7);

  // Any nullary callable whose declared result is `nullval` works as a terminating action.
  const auto bail = []() noexcept -> nullval {
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

  // `defer` captures instead of reacting: the callable supplies an empty pointer on success;
  // a throw yields the active exception, ready for a later rethrow — non-std included.
  const ::std::exception_ptr clean = on_throw(defer) << [] {
    return ::std::exception_ptr{};
  };
  EXPECT(!clean);
  const ::std::exception_ptr held = on_throw(defer) << []() -> ::std::exception_ptr {
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
  const ::std::exception_ptr odd = on_throw(defer) << []() -> ::std::exception_ptr {
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

  // catch_exactly is monomorphic: the exact dynamic type handles, everything else declines.
  const int rx1 = on_throw(catch_exactly<::std::logic_error>(subst(1)) | subst(2)) << []() -> int {
    throw ::std::logic_error{"exact"};
  };
  EXPECT(rx1 == 1);
  const int rx2 = on_throw(catch_exactly<::std::logic_error>(subst(1)) | subst(2)) << []() -> int {
    throw ::std::domain_error{"derived, so no exact match"};
  };
  EXPECT(rx2 == 2);
  const int rx3 = on_throw(catch_exactly<::std::logic_error>(subst(1)) | subst(2)) << []() -> int {
    throw 42; // non-std: the funnel is null, catch_exactly must decline
  };
  EXPECT(rx3 == 2);
  // Layered severity: the exact type recovers, the rest of its cone takes the next arm.
  const int rx4 = on_throw(catch_exactly<::std::logic_error>(subst(1)) | catch_only<::std::logic_error>(subst(2)))
               << []() -> int {
                    throw ::std::domain_error{"cone remainder"};
                  };
  EXPECT(rx4 == 2);

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
  //      -> "the left type guard already claims every exception type the right arm lists; ..."
  //  5b. on_throw(catch_only<std::logic_error>(subst(1)) | catch_exactly<std::logic_error>(subst(2))) << ...
  //      -> same message: the cone on the left starves the exact entry inside it
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
  const ::std::exception_ptr np = on_throw(notify(noted) & defer) << []() -> ::std::exception_ptr {
    throw ::std::runtime_error("noted+deferred");
  };
  EXPECT(!!np);
  EXPECT(noted.str().find("noted+deferred") != ::std::string::npos);

  // as_expected adapts to the callable's declared expected type on both paths.
  using Result   = ::cuda::std::expected<int, ::std::exception_ptr>;
  const auto good = on_throw(as_expected) << []() -> Result {
    return 5; // expected's converting constructor keeps the happy path natural
  };
  static_assert(::cuda::std::is_same_v<decltype(good), const Result>,
                "the callable owns the as_expected expression type");
  EXPECT(good.has_value());
  // Dereference rather than .value(): value() would instantiate bad_expected_access, whose
  // inlined exception_ptr destructor trips a spurious gcc 14/15 -O3 maybe-uninitialized in
  // every TU that compiles these tests.
  EXPECT(*good == 5);

  const auto bad = on_throw(as_expected) << []() -> Result {
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

  // Re-attempt success preserves the callable's declared boundary type.
  {
    using Result = ::cuda::std::expected<int, ::std::exception_ptr>;
    int calls     = 0;
    const auto r  = on_throw(as_expected & retry) << [&]() -> Result {
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
    const auto ep = on_throw(defer & retry) << [&]() -> ::std::exception_ptr {
      if (++calls < 2)
      {
        throw ::std::runtime_error("once");
      }
      return {};
    };
    EXPECT(!ep); // empty: the re-attempt succeeded and the callable supplied the value
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
  //  - on_throw(as_expected) << []() -> int { ... };
  //      -> "as_expected requires the callable to return a cuda::std::expected instantiation"
#  endif // CCCL_HAS_EXCEPTIONS()
};

// Helper exception types for the tests below, at namespace scope on purpose: nvcc <= 12.9 in
// C++20 mode infers host__ device__ for a local class's special members inside an extended
// lambda, and the inherited std::runtime_error constructor is host-only (error #20011).
struct ut_error_from_exception
{
  int marker;

  explicit ut_error_from_exception(const ::std::exception& exception)
      : marker(exception.what()[0])
  {}
};
struct ut_low_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};
struct ut_high_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};

UNITTEST("as_expected and defer")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if CCCL_HAS_EXCEPTIONS()
  using PtrResult = ::cuda::std::expected<int, ::std::exception_ptr>;
  using RefResult = ::cuda::std::expected<int, ut_error_from_exception>;
  static_assert(detail::is_expected<PtrResult>);
  static_assert(detail::is_expected<RefResult>);
  static_assert(!detail::is_expected<int>);

  // The callable declares the boundary type; expected's converting constructor keeps a bare
  // success return natural.
  {
    const auto r = on_throw(as_expected) << []() -> PtrResult {
      return 42;
    };
    static_assert(::cuda::std::is_same_v<decltype(r), const PtrResult>);
    EXPECT(r.has_value());
    EXPECT(*r == 42);
  }

  // First ladder rung: the error type accepts the active exception_ptr.
  {
    const auto r = on_throw(as_expected) << []() -> PtrResult {
      throw ::std::runtime_error("captured");
    };
    EXPECT(!r.has_value());
    bool rethrown = false;
    try
    {
      ::std::rethrow_exception(r.error());
    }
    catch (const ::std::runtime_error& exception)
    {
      rethrown = ::std::string_view{exception.what()} == "captured";
    }
    EXPECT(rethrown);
  }

  // Second ladder rung: construct the error from the funneled std::exception.
  {
    const auto r = on_throw(as_expected) << []() -> RefResult {
      throw ::std::runtime_error("reference");
    };
    EXPECT(!r.has_value());
    EXPECT(r.error().marker == 'r');
  }

  // A nonstandard exception cannot use the std::exception rung, so it declines unchanged.
  {
    bool escaped = false;
    try
    {
      on_throw(as_expected) << []() -> RefResult {
        throw 42;
      };
    }
    catch (int)
    {
      escaped = true;
    }
    EXPECT(escaped);
  }

  // defer uses the same callable-owned type: empty on success, active pointer on failure.
  {
    const ::std::exception_ptr clean = on_throw(defer) << [] {
      return ::std::exception_ptr{};
    };
    EXPECT(!clean);
    const ::std::exception_ptr held = on_throw(defer) << []() -> ::std::exception_ptr {
      throw ::std::logic_error("deferred");
    };
    EXPECT(!!held);
    bool rethrown = false;
    try
    {
      ::std::rethrow_exception(held);
    }
    catch (const ::std::logic_error& exception)
    {
      rethrown = ::std::string_view{exception.what()} == "deferred";
    }
    EXPECT(rethrown);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(as_expected) << []() -> int { return 1; };
  //      -> "as_expected requires the callable to return a cuda::std::expected instantiation"
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
      on_throw(when(
        [] {
          return true;
        },
        subst(3)))
      << []() -> int {
      throw ut_low_error("low");
    };
    const int declined =
      on_throw(when(
                 [] {
                   return false;
                 },
                 subst(3))
               | subst(4))
      << []() -> int {
      throw ut_low_error("low");
    };
    EXPECT(accepted == 3);
    EXPECT(declined == 4);
  }

  // translate<From, To>: a From becomes a To for the next typed arm; non-From declines.
  {
    const int v = on_throw(translate<ut_low_error, ut_high_error> | catch_only<ut_high_error>(subst(1)))
               << []() -> int {
      throw ut_low_error("cause");
    };
    EXPECT(v == 1);
    const int passed = on_throw(translate<ut_low_error, ut_high_error> | subst(2)) << []() -> int {
      throw ::std::runtime_error("neither");
    };
    EXPECT(passed == 2);
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
    const int got = on_throw(remember(&last)) << [] {
      return 7;
    };
    EXPECT(got == 7);
    EXPECT(last == 7);

    const int stale = on_throw(remember(&last)) << []() -> int {
      throw ut_low_error("offline");
    };
    EXPECT(stale == 7);

    const int fresh = ON_THROW(remember(&last))
    {
      return 9;
    };
    const int served = ON_THROW(remember(&last))->int
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
    int& fresh        = on_throw(remember(&last)) << [&]() -> int& {
      return source;
    };
    int& stale = on_throw(remember(&last)) << []() -> int& {
      throw ut_low_error("offline");
    };
    EXPECT(&fresh == &source);
    EXPECT(last == 11);
    EXPECT(&stale == &last);
  }

  // remember over a shared cell: the policy co-owns it.
  {
    auto cell     = ::std::make_shared<int>(0);
    const int got = on_throw(remember(cell)) << [] {
      return 21;
    };
    EXPECT(got == 21);
    EXPECT(*cell == 21);
    const int stale = on_throw(remember(cell)) << []() -> int {
      throw ::std::runtime_error("offline");
    };
    EXPECT(stale == 21);
  }

  // always: every element runs on both paths; the original exception survives.
  {
    int notes = 0;
    auto note = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++notes;
    };
    const int ok = on_throw(always(subst(1), note) | subst(2)) << []() -> int {
      throw ::std::runtime_error("x");
    };
    EXPECT(ok == 1); // subst accepted; note also ran
    EXPECT(notes == 1);
  }
  {
    int notes = 0;
    auto note = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++notes;
    };
    bool escaped = false;
    try
    {
      on_throw(always(rethrow, note)) << []() -> int {
        throw ::std::runtime_error("orig");
      };
    }
    catch (const ::std::runtime_error& e)
    {
      escaped = ::std::string_view{e.what()} == "orig";
    }
    EXPECT(escaped); // the head declined; note still ran; the ORIGINAL propagated
    EXPECT(notes == 1);
  }
  {
    int first = 0, second = 0;
    auto f = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++first;
    };
    auto g = [&](const ::std::exception*, const ::cuda::std::source_location, auto&) {
      ++second;
    };
    bool escaped = false;
    try
    {
      on_throw(always(rethrow, f, g)) << []() -> int {
        throw ::std::runtime_error("x");
      };
    }
    catch (...)
    {
      escaped = true;
    }
    EXPECT(escaped); // variadic: both finalizers ran despite the head declining
    EXPECT(first == 1);
    EXPECT(second == 1);
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

  // Success mid-repetition returns the callable's declared boundary type.
  {
    using Result = ::cuda::std::expected<int, ::std::exception_ptr>;
    int calls     = 0;
    const auto r  = on_throw((as_expected & retry) * 3) << [&]() -> Result {
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

UNITTEST("type erasure")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
#  if CCCL_HAS_EXCEPTIONS()
  const auto names_both = [](const ::std::logic_error& err, ::std::string_view a, ::std::string_view b) {
    const ::std::string msg{err.what()};
    return msg.find(::std::string{a}) != ::std::string::npos
        && msg.find(::std::string{b}) != ::std::string::npos;
  };

  // Universality: retry * 3 erased generically -- internal loop, state,
  // and rethrow-on-exhaustion preserved. Retry-through is passthrough/unchecked.
  {
    int calls   = 0;
    const int x = on_throw(type_erase(retry * 3)) << [&]() -> int {
      if (++calls < 3)
      {
        throw ::std::runtime_error("flaky");
      }
      return 42;
    };
    EXPECT(x == 42);
    EXPECT(calls == 3);
  }
  // Exhaustion rethrows into the next arm; the callable owns the result type.
  {
    int calls   = 0;
    const int x = on_throw(type_erase(retry * 3) | subst(-7)) << [&]() -> int {
      ++calls;
      throw ::std::runtime_error("always");
    };
    EXPECT(x == -7);
    EXPECT(calls == 4); // one initial attempt + three retries
  }
  // One sink object serves callables of different result types (passthrough).
  {
    exception_sink r = type_erase(retry * 2);
    {
      int calls   = 0;
      const int x = on_throw(r) << [&]() -> int {
        if (++calls < 2)
        {
          throw ::std::runtime_error("flaky");
        }
        return 5;
      };
      EXPECT(x == 5);
    }
    {
      int calls             = 0;
      const ::std::string x = on_throw(r) << [&]() -> ::std::string {
        if (++calls < 2)
        {
          throw ::std::runtime_error("flaky");
        }
        return "ok";
      };
      EXPECT(x == "ok");
    }
  }
  // int stored under long body: the body's range contains the answer.
  {
    const long x = on_throw(type_erase(subst(9))) << []() -> long {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 9L);
  }
  // long long stored under int body: fail eagerly (portable stand-in for a strictly wider integral).
  {
    bool failed = false;
    try
    {
      on_throw(type_erase(subst(9LL))) << []() -> int {
        throw ::std::runtime_error("x");
      };
    }
    catch (const ::std::logic_error& err)
    {
      failed = names_both(err, type_name<long long>, type_name<int>);
    }
    EXPECT(failed);
  }
  // int under double: integral answers pass under a floating body.
  {
    const double x = on_throw(type_erase(subst(9))) << []() -> double {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 9.0);
  }
  // double under float: precision loss is tolerated.
  {
    const float x = on_throw(type_erase(subst(1.5))) << []() -> float {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == static_cast<float>(1.5));
  }
  // double under int: floating never converts to integral; fail eagerly.
  {
    bool failed = false;
    try
    {
      on_throw(type_erase(subst(1.5))) << []() -> int {
        throw ::std::runtime_error("x");
      };
    }
    catch (const ::std::logic_error& err)
    {
      failed = names_both(err, type_name<double>, type_name<int>);
    }
    EXPECT(failed);
  }
  // defer under int: fail on first use, naming both types.
  {
    bool failed = false;
    try
    {
      on_throw(type_erase(defer)) << []() -> int {
        return 1;
      };
    }
    catch (const ::std::logic_error& err)
    {
      failed = names_both(err, type_name<::std::exception_ptr>, type_name<int>);
    }
    EXPECT(failed);
  }
  // Resume / effects: exempt from the type check. Resume over void is legal.
  {
    int hits = 0;
    on_throw(type_erase(::std::ignore)) << [&]() -> void {
      ++hits;
      throw ::std::runtime_error("x");
    };
    EXPECT(hits == 1);
  }
  {
    const int x = on_throw(type_erase(::std::ignore)) << []() -> int {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 0);
  }
  // Metadata: const fields, derived at erasure time.
  EXPECT(type_erase(retry * 3).kind() == exception_sink::answer_kind::passthrough);
  EXPECT(type_erase(subst(9)).kind() == exception_sink::answer_kind::integral);
  EXPECT(type_erase(subst(1.5)).kind() == exception_sink::answer_kind::floating);
  EXPECT(type_erase(::std::ignore).kind() == exception_sink::answer_kind::resumes);
  EXPECT((type_erase(translate<::std::runtime_error, ::std::logic_error>).kind() == exception_sink::answer_kind::dies));
  EXPECT(type_erase(retry * 3).may_rethrow());
  EXPECT(!type_erase(::std::ignore).may_rethrow());
  // Re-erasure: passthrough composite; first-throw unbox is the backstop.
  {
    const int x = on_throw(type_erase(type_erase(subst(5)))) << []() -> int {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 5);
  }
  // Dynamic and static policies side by side in one expression.
  {
    const int x =
      on_throw(when(
        [] {
          return true;
        },
        type_erase(subst(11))))
      << []() -> int {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 11);
  }
  // The success path delivers the callable's result, unboxed to the same type.
  {
    const int x = on_throw(type_erase(subst(1))) << []() -> int {
      return 30; // no throw
    };
    EXPECT(x == 30);
  }
  // A custom model derived from the public sink_base defaults to passthrough/unchecked.
  {
    struct halving_sink final : exception_sink::sink_base
    {
      halving_sink()
          : sink_base(exception_sink::answer_kind::passthrough, false)
      {}
      sink_base* clone() const override
      {
        return new halving_sink();
      }
      ::std::any hook(const ::std::exception*, ::cuda::std::source_location, ::std::function<::std::any()>&) override
      {
        return ::std::any(21);
      }
    };
    exception_sink custom{::std::unique_ptr<exception_sink::sink_base>(new halving_sink())};
    const int x = on_throw(custom) << []() -> int {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 21);
    EXPECT(!custom.may_rethrow());
    EXPECT(custom.kind() == exception_sink::answer_kind::passthrough);
  }

  // The passthrough backstop applies the conversion law per VALUE at first throw. Only
  // |-composites erase as passthrough (they interpret internally at std::any); a pure-&
  // composite keeps its final policy's precise kind and is checked at first use instead.
  {
    // A stored int that FITS the unsigned body converts.
    const unsigned x =
      on_throw(type_erase(
        when(
          [] {
            return true;
          },
          subst(7))
        | subst(0u)))
      << []() -> unsigned {
      throw ::std::runtime_error("x");
    };
    EXPECT(x == 7u);
  }
  {
    // A stored -1 under an unsigned body is rejected loudly, not wrapped.
    bool failed = false;
    try
    {
      on_throw(type_erase(
        when(
          [] {
            return true;
          },
          subst(-1))
        | subst(0)))
        << []() -> unsigned {
        throw ::std::runtime_error("x");
      };
    }
    catch (const ::std::logic_error& err)
    {
      failed = ::std::string_view{err.what()}.find(type_name<unsigned>) != ::std::string_view::npos;
    }
    EXPECT(failed);
  }
  {
    // A stored floating value never lands in an integral body, even through the backstop.
    bool failed = false;
    try
    {
      on_throw(type_erase(
        when(
          [] {
            return true;
          },
          subst(1.5))
        | subst(0)))
        << []() -> int {
        throw ::std::runtime_error("x");
      };
    }
    catch (const ::std::logic_error& err)
    {
      failed = ::std::string_view{err.what()}.find(type_name<int>) != ::std::string_view::npos;
    }
    EXPECT(failed);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  1. on_throw(as_expected) << []() -> int { return 1; };
  //       -> "as_expected requires the callable to return a cuda::std::expected instantiation"
  //  2. a policy whose on_success returns a different type than decltype(fn());
  //       -> "a policy's on_success must preserve the expression type; policies no longer own it (SPEC-ADDENDUM-7)"
  //  3. on_throw(type_erase(subst(1))) << []() -> int& { static int x = 0; return x; };
  //       -> "exception_sink cannot serve a reference-returning callable: std::any cannot carry references; use a
  //       concrete policy, or return a pointer"
  //  4. on_throw(subst(-1)) << []() -> unsigned { throw 0; };
  //       -> "the policy's answer does not preserve the callable's value range ...; write the conversion in the
  //       policy -- subst(0xffffffffu), not subst(-1) -- if the narrowing is intended"
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
