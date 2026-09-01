//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief SCOPE guards and exception handling (`on_throw`)
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/std/expected>
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
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/underlying_type.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__stf/utility/source_location.cuh>
#include <cuda/experimental/__stf/utility/traits.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

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
  using __exception_sink_tag = void;
  //! @endcond

  // Destination: `__os_` wins if set, otherwise `__file_` (default `stderr`).
  ::FILE* __file_       = stderr;
  ::std::ostream* __os_ = nullptr;

  //! @brief Returns a copy configured to report on `__file` instead of `stderr`.
  notify_t operator()(::FILE* __file) const
  {
    notify_t __copy;
    __copy.__file_ = __file;
    return __copy;
  }

  //! @brief Returns a copy configured to report on `__os` instead of `stderr`.
  notify_t operator()(::std::ostream& __os) const
  {
    notify_t __copy;
    __copy.__os_ = &__os;
    return __copy;
  }

  //! @brief Reports the exception and resumes. Writes to the configured `std::ostream` if any,
  //! else `__file_` (default `stderr`). The ostream write is best-effort: a stream configured
  //! to throw does not get to end the program from inside a handler.
  template <class _Fn>
  decltype(::std::ignore)
  operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn&) const noexcept
  {
    if (__os_)
    {
      _CCCL_TRY
      {
        *__os_ << __loc.file_name() << '(' << __loc.line() << ") on_throw violation in " << __loc.function_name()
               << ": " << (__exception ? __exception->what() : "nonstandard exception") << '\n';
        __os_->flush();
      }
      _CCCL_CATCH_ALL {}
    }
    else
    {
      ::fprintf(__file_,
                "%s(%u) on_throw violation in %s: %s\n",
                __loc.file_name(),
                __loc.line(),
                __loc.function_name(),
                __exception ? __exception->what() : "nonstandard exception");
      ::fflush(__file_);
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
  using __exception_sink_tag = void;
  //! @endcond

  //! @brief The exception hook: report, then die.
  template <class _Fn>
  [[noreturn]] nullval
  operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn) const noexcept
  {
    notify(__exception, __loc, __fn);
    ::std::abort();
  }
};
inline constexpr abort_t abort{};

//! @brief Like @ref abort_t "abort", but ends via `std::terminate`.
struct terminate_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  template <class _Fn>
  [[noreturn]] nullval
  operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn) const noexcept
  {
    notify(__exception, __loc, __fn);
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
  using __exception_sink_tag = void;
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
 *
 * Inside a catch block, `on_throw(defer)` captures the exception thrown by its own guarded
 * body (the newest exception, current inside its own catch), never the exception the
 * surrounding handler is handling; for that one, call `std::current_exception()` directly.
 */
struct defer_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  //! @brief Captures the in-flight exception; the answer converts to the expression's type.
  template <class _Fn>
  ::std::exception_ptr operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const noexcept
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
  using __exception_sink_tag = void;
  //! @endcond

  template <class _Fn>
  [[noreturn]] nullval operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const
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
template <class _V>
struct subst_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  _V __v_;

  template <class _Fn>
  decltype(auto) operator()([[maybe_unused]] const ::std::exception* __exception,
                            [[maybe_unused]] const ::cuda::std::source_location __loc,
                            [[maybe_unused]] _Fn& __fn) noexcept
  {
    if constexpr (::cuda::std::is_invocable_v<_V&, const ::std::exception*, ::cuda::std::source_location, _Fn&>)
    {
      return __v_(__exception, __loc, __fn);
    }
    else if constexpr (::cuda::std::is_invocable_v<_V&>)
    {
      return __v_();
    }
    else
    {
      return ::cuda::std::forward<_V>(__v_);
    }
  }
};

//! @brief Creates a value-substitution policy; see @ref subst_t.
template <class _V>
auto subst(_V&& __v)
{
  return subst_t<_V>{::cuda::std::forward<_V>(__v)};
}

/**
 * @brief The unit re-attempt: re-runs the callable once; if the re-run throws, declines with
 * that exception. Counts come from repetition: `retry * 3` re-attempts up to three times;
 * `retry * 3 | subst(fallback)` answers the spent failure. `retry` alone is `retry * 1`.
 *
 * Answers `__fn()`'s value for a non-void callable, or `std::ignore` after a successful
 * re-run of a void callable. `&` discards non-final answers, so `retry & subst(0)` is legal
 * (and almost never what you want).
 */
struct retry_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  // decltype(auto), not auto: a reference-returning callable must re-run to the same object,
  // not to a copy (the ignore branch deduces the object type and returns a copy of the tag).
  template <class _Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn& __fn)
  {
    if constexpr (::cuda::std::is_void_v<decltype(__fn())>)
    {
      __fn(); // a throw here IS the decline
      return ::std::ignore; // resume: the void expression is complete
    }
    else
    {
      return __fn();
    }
  }
};
inline constexpr retry_t retry{};

namespace detail
{
template <class>
inline constexpr bool __is_expected = false;

template <class _T, class _E>
inline constexpr bool __is_expected<::cuda::std::expected<_T, _E>> = true;

template <class>
struct __expected_error;

template <class _T, class _E>
struct __expected_error<::cuda::std::expected<_T, _E>>
{
  using type = _E;
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
  using __exception_sink_tag = void;

  template <class _Fn, class _Raw = decltype(::cuda::std::declval<_Fn&>()())>
  auto operator()(const ::std::exception* __exception, const ::cuda::std::source_location, _Fn&) const -> _Raw
  {
    // nvcc instantiates this body when forming the callable-independent presence probe
    // (`void (&)()`). Keep that archetype admissible; the assert still fires at a real
    // composition site (a user void-lambda is a distinct type, not `void()`).
    if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Fn>, void()>)
    {
      _CCCL_UNREACHABLE();
    }
    else
    {
      using _Expected = ::cuda::std::remove_cvref_t<_Raw>;
      static_assert(detail::__is_expected<_Expected>,
                    "as_expected requires the callable to return a cuda::std::expected instantiation");

      if constexpr (detail::__is_expected<_Expected>)
      {
        using _E = typename detail::__expected_error<_Expected>::type;
        if constexpr (::cuda::std::is_constructible_v<_E, ::std::exception_ptr>)
        {
          return _Raw{::cuda::std::unexpect, _E(::std::current_exception())};
        }
        else if constexpr (::cuda::std::is_constructible_v<_E, const ::std::exception&>)
        {
          if (__exception)
          {
            return _Raw{::cuda::std::unexpect, _E(*__exception)};
          }
          throw; // nonstandard exception, no lossless construction rung: decline
        }
        else
        {
          static_assert(
            ::cuda::std::is_constructible_v<_E, ::std::exception_ptr>
              || ::cuda::std::is_constructible_v<_E, const ::std::exception&>,
            "as_expected requires the error type constructible from exception_ptr or const std::exception&");
        }
      }
      _CCCL_UNREACHABLE();
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
template <class _Pred>
struct when_t
{
  using __exception_sink_tag = void;
  _Pred __pred_;

  template <class _Fn>
  // maybe_unused: when the predicate is nullary, only the discarded constexpr
  // branch reads __exception; gcc 9 reports it as set-but-unused.
  void operator()([[maybe_unused]] const ::std::exception* __exception, const ::cuda::std::source_location, _Fn&)
  {
    if constexpr (::cuda::std::is_invocable_v<_Pred&, const ::std::exception*>)
    {
      if (__pred_(__exception))
      {
        return;
      }
    }
    else
    {
      if (__pred_())
      {
        return;
      }
    }
    throw; // decline: the guard does not apply
  }
};

/**
 * @brief Boundary translation: catches a `_From` (catch-clause rules: same or publicly
 * derived) and throws a `_To` -- constructed from the caught `_From` when such a constructor
 * exists, default-constructed otherwise. Anything that is not a `_From` declines untouched,
 * so `translate<low, high> | ...` ladders compose; a following arm sees the `_To`.
 */
template <class _From, class _To>
struct translate_t
{
  using __exception_sink_tag = void;

  template <class _Fn>
  [[noreturn]] nullval operator()(const ::std::exception* __e, const ::cuda::std::source_location, _Fn&) const
  {
    if (__e)
    {
      // The funnel pointer decides for std-derived exceptions, no rethrow needed.
      if (const auto* __from = dynamic_cast<const _From*>(__e))
      {
        __throw_translated(*__from);
      }
      throw; // decline: a std exception that is not a _From
    }
    // A non-std exception: re-observe at _From.
    _CCCL_TRY
    {
      throw;
    }
    _CCCL_CATCH (const _From& __from)
    {
      __throw_translated(__from);
    }
    _CCCL_CATCH_FALLTHROUGH // decline: not a _From either
    _CCCL_UNREACHABLE();
  }

private:
  [[noreturn]] static void __throw_translated(const _From& __from)
  {
    if constexpr (::cuda::std::is_constructible_v<_To, const _From&>)
    {
      throw _To(__from);
    }
    else if constexpr (::cuda::std::is_base_of_v<::std::exception, _From>
                       && ::cuda::std::is_constructible_v<_To, const char*>)
    {
      throw _To(__from.what()); // carry the message across the translation
    }
    else if constexpr (::cuda::std::is_default_constructible_v<_To>)
    {
      throw _To{};
    }
    else
    {
      static_assert(!::cuda::std::is_same_v<_From, _From>,
                    "translate<From, To>: To must be constructible from const From&, from "
                    "From::what(), or default-constructible");
    }
  }
};

//! @brief Translates: catches a `_From` (catch-clause rules), throws a `_To` -- constructed from
//! the caught `_From` when possible, default-constructed otherwise. Anything else declines.
template <class _From, class _To>
inline constexpr translate_t<_From, _To> translate{};

/**
 * @brief Throws a stored exception with the active exception nested as its cause.
 */
template <class _E>
struct nest_t
{
  using __exception_sink_tag = void;
  _E __exception_;

  static_assert(::cuda::std::is_copy_constructible_v<_E>, "nest(e) requires a copyable exception object");

  template <class _Fn>
  [[noreturn]] nullval operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&)
  {
    _CCCL_TRY
    {
      throw;
    }
    _CCCL_CATCH_ALL
    {
      ::std::throw_with_nested(__exception_);
    }
    _CCCL_UNREACHABLE();
  }
};

//! @brief Stores an exception by value and nests the active exception beneath it.
template <class _E>
auto nest(_E&& __exception)
{
  using _Stored = ::cuda::std::decay_t<_E>;
  return nest_t<_Stored>{::cuda::std::forward<_E>(__exception)};
}

/**
 * @brief Effect policy that sleeps before the next element of an `&` sequence.
 *
 * `(delay(100ms) & retry) * 3` pauses before each re-attempt.
 */
template <class _Duration>
struct delay_t
{
  using __exception_sink_tag = void;
  _Duration __duration_;

  template <class _Fn>
  void operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&)
  {
    ::std::this_thread::sleep_for(__duration_);
  }
};

//! @brief Creates a sleeping effect policy; see @ref delay_t.
template <class _Duration>
auto delay(_Duration&& __duration)
{
  return delay_t<_Duration>{::cuda::std::forward<_Duration>(__duration)};
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
  using __exception_sink_tag = void;
  int __n_;
  ::std::chrono::milliseconds __initial_;

  template <class _Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn& __fn)
  {
    if (__n_ == 0)
    {
      throw;
    }

    const auto __base = __initial_.count();
    // maybe_unused: referenced only inside _CCCL_CATCH_ALL, which the device pass
    // expands to a discarded branch; CTK <= 12.9's cudafe then reports #177.
    [[maybe_unused]] const auto __cap = __base * 64;
    auto __sleep                      = __base;
    auto __state = static_cast<unsigned long long>(::std::chrono::steady_clock::now().time_since_epoch().count());
    if (__state == 0)
    {
      __state = 1;
    }

    // maybe_unused: like __cap above, __left is referenced only inside
    // _CCCL_CATCH_ALL, so CTK <= 12.9's cudafe reports #177 without it.
    for ([[maybe_unused]] int __left = __n_;;)
    {
      ::std::this_thread::sleep_for(::std::chrono::milliseconds{__sleep});
      _CCCL_TRY
      {
        if constexpr (::cuda::std::is_void_v<decltype(__fn())>)
        {
          __fn();
          return ::std::ignore;
        }
        else
        {
          return __fn();
        }
      }
      _CCCL_CATCH_ALL
      {
        if (--__left == 0)
        {
          throw;
        }
        __state ^= __state << 13;
        __state ^= __state >> 7;
        __state ^= __state << 17;
        const auto __tripled = __sleep * 3;
        const auto __upper   = __tripled < __cap ? __tripled : __cap;
        const auto __span    = __upper - __base + 1;
        __sleep = __base + static_cast<decltype(__base)>(__state % static_cast<unsigned long long>(__span));
      }
    }
    _CCCL_UNREACHABLE();
  }
};

//! @brief Creates a decorrelated-jitter retry policy.
inline backoff_t backoff(int __n, ::std::chrono::milliseconds __initial)
{
  _CCCL_ASSERT(__n >= 0, "backoff requires a non-negative retry count");
  return backoff_t{__n, __initial};
}

/**
 * @brief Serves the last successful value when a later call throws.
 *
 * The cell is designated by pointer -- a raw `T*` for a caller-owned variable, or a
 * `shared_ptr<T>` when the policy should share ownership; the two are distinguished
 * statically by overload. Success updates the cell and passes the result through; failure
 * substitutes the stored value (an lvalue: it can serve reference results).
 */
template <class _Ptr>
struct remember_t
{
  using __exception_sink_tag = void;
  _Ptr __cell_; // a raw pointer or a shared_ptr; *__cell_ is the last-known-good value

  template <class _R>
  ::cuda::std::conditional_t<::cuda::std::is_lvalue_reference_v<_R&&>, _R&&, ::cuda::std::remove_cvref_t<_R>>
  on_success(_R&& __result)
  {
    *__cell_ = __result;
    return ::cuda::std::forward<_R>(__result);
  }

  template <class _Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const noexcept
  {
    return *__cell_;
  }
};

//! @brief Creates a last-known-good policy over a caller-owned cell.
template <class _T>
auto remember(_T* __cell)
{
  _CCCL_ASSERT(__cell, "remember requires a non-null cell");
  return remember_t<_T*>{__cell};
}

//! @brief Creates a last-known-good policy that shares ownership of its cell.
template <class _T>
auto remember(::std::shared_ptr<_T> __cell)
{
  _CCCL_ASSERT(__cell, "remember requires a non-null cell");
  return remember_t<::std::shared_ptr<_T>>{::cuda::std::move(__cell)};
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
  using __exception_sink_tag = void;
  //! @endcond

  ::std::shared_ptr<int> __budget_;
  int __initial_;

  //! @brief The entry gate: refuses the attempt once the budget is spent.
  void on_enter() const
  {
    if (*__budget_ <= 0)
    {
      throw circuit_open{};
    }
  }

  //! @brief The exception hook: record the failure, answer as an effect.
  template <class _Fn>
  decltype(::std::ignore) operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const noexcept
  {
    --*__budget_;
    return ::std::ignore;
  }

  //! @brief Success restores the budget to its creation-time value.
  template <class _R>
  ::cuda::std::conditional_t<::cuda::std::is_lvalue_reference_v<_R&&>, _R&&, ::cuda::std::remove_cvref_t<_R>>
  on_success(_R&& __result) const
  {
    *__budget_ = __initial_;
    return ::cuda::std::forward<_R>(__result);
  }

  void on_success() const
  {
    *__budget_ = __initial_;
  }
};

//! @brief Creates a counter-based circuit breaker; see @ref circuit_breaker_t. The initial
//! `*__budget` is the failure allowance restored on success.
inline circuit_breaker_t circuit_breaker(::std::shared_ptr<int> __budget)
{
  _CCCL_ASSERT(__budget, "circuit_breaker requires a non-null budget");
  const int __initial = *__budget;
  return circuit_breaker_t{::std::move(__budget), __initial};
}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// --- The policy protocol: two optional capabilities discovered by introspection ------------
//
// Each capability is one archetype alias + `_IsValidExpansion`. The exception hook is always
// `(const std::exception*, source_location, Fn&)`; Fn-independent probes use a throwaway
// `void (&)()`. `__hook_answer_t` IS the hook archetype -- one definition, both roles.

template <class _P, class _Fn>
using __exception_hook_of = decltype(::cuda::std::declval<_P&>()(
  ::cuda::std::declval<const ::std::exception*>(),
  ::cuda::std::declval<::cuda::std::source_location>(),
  ::cuda::std::declval<_Fn&>()));

template <class _P, class _Fn = void (&)()>
inline constexpr bool __has_exception_hook =
  ::cuda::std::_IsValidExpansion<__exception_hook_of, ::cuda::std::remove_reference_t<_P>, _Fn>::value;

template <class _P, class _Fn = void (&)()>
using __hook_answer_t = __exception_hook_of<::cuda::std::remove_reference_t<_P>, _Fn>;

// Capability 2a: the success hook `p.on_success(R&&)` for a given result type.
template <class _P, class _R>
using __on_success_with_of = decltype(::cuda::std::declval<_P&>().on_success(::cuda::std::declval<_R>()));

template <class _P, class _R>
inline constexpr bool __has_on_success_with =
  ::cuda::std::_IsValidExpansion<__on_success_with_of, ::cuda::std::remove_reference_t<_P>, _R>::value;

// Capability 2b: the nullary success hook `p.on_success()` (the void-result channel).
template <class _P>
using __on_success_void_of = decltype(::cuda::std::declval<_P&>().on_success());

template <class _P>
inline constexpr bool __has_on_success_void =
  ::cuda::std::_IsValidExpansion<__on_success_void_of, ::cuda::std::remove_reference_t<_P>>::value;

// Capability 2c: the entry hook `p.on_enter()`, run once per attempt expression, before the
// callable and outside the policy's own catch. It answers nothing: it either admits the
// attempt or refuses it by throwing.
template <class _P>
using __on_enter_of = decltype(::cuda::std::declval<_P&>().on_enter());

template <class _P>
inline constexpr bool __has_on_enter =
  ::cuda::std::_IsValidExpansion<__on_enter_of, ::cuda::std::remove_reference_t<_P>>::value;

// Nothrow-ness of the entry hook, vacuously true when absent (two-step form: `&&` does not
// short-circuit template instantiation).
template <class _P, bool = __has_on_enter<_P>>
inline constexpr bool __on_enter_nothrow_v = true;

template <class _P>
inline constexpr bool __on_enter_nothrow_v<_P, true> =
  noexcept(::cuda::std::declval<::cuda::std::remove_reference_t<_P>&>().on_enter());

// A policy is anything exposing at least one capability (exception hook probed with a throwaway).
template <class _P>
inline constexpr bool __has_any_capability = __has_exception_hook<_P> || __has_on_success_void<_P>;

// Whether a type is one this header defines as an exception-sink policy, marked by the
// __exception_sink_tag member. This is what &/| require of at least one operand, so they do
// not hijack unrelated types.
template <class _P>
using __exception_sink_tag_of = typename _P::__exception_sink_tag;

template <class _P>
inline constexpr bool __is_exception_sink_v =
  ::cuda::std::_IsValidExpansion<__exception_sink_tag_of, ::cuda::std::remove_cvref_t<_P>>::value;

// Whether a reaction is `::std::ignore` (compared as the current code does).
template <class _P>
inline constexpr bool __is_ignore_v =
  ::cuda::std::is_same_v<const ::cuda::std::remove_cvref_t<_P>,
                         const ::cuda::std::remove_reference_t<decltype(::std::ignore)>>;

// Whether a policy's answer type is `nullval` -- it never returns from the exception path. The
// two-step form keeps `__hook_answer_t` from being named for a hookless policy: `&&` does not
// short-circuit template instantiation, so the answer is probed only in the `true` partial.
template <bool _HasHook, class _P, class _Fn>
inline constexpr bool __answers_nothing_impl = false;

template <class _P, class _Fn>
inline constexpr bool __answers_nothing_impl<true, _P, _Fn> =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<__hook_answer_t<_P, _Fn>>, nullval>;

template <class _P, class _Fn = void (&)()>
inline constexpr bool __answers_nothing = __answers_nothing_impl<__has_exception_hook<_P, _Fn>, _P, _Fn>;

// Whether a policy's exception path is nothrow -- the same computation that
// `operator<<`'s conditional noexcept uses. Declining from `|` means throwing
// from that path, so a nothrow left side of `|` leaves the right unreachable.
template <class _P, class _Fn = void (&)()>
inline constexpr bool __exception_path_nothrow_v =
  __has_exception_hook<_P, _Fn>
  && ::cuda::std::is_nothrow_invocable_v<::cuda::std::remove_reference_t<_P>&,
                                         const ::std::exception*,
                                         ::cuda::std::source_location,
                                         _Fn&>;

// --- Adapters: normalize the historical reactions into policies ----------------------------

// `::std::ignore` as a policy: resume with a default-constructed result. Resume is
// polymorphic substitution of the default: on non-void callables this policy is equivalent to
// `subst([](auto*, auto, auto& fn) { return decltype(fn())(); })`; the marker exists because a
// void expression has no value to substitute, yet "resumed" and "merely an effect" must stay
// distinguishable answers.
struct __ignore_policy
{
  using __exception_sink_tag = void;

  template <class _Fn>
  decltype(::std::ignore) operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const noexcept
  {
    return ::std::ignore;
  }
};

// Success-hook forwarding shared by the single-policy wrappers (`__catch_only_t`,
// `__as_policy`, `__policy_pow`): both arities delegate to the wrapped policy. The outer
// `operator<<` enforces that the forwarded answer preserves the callable's expression type.
template <class _P>
struct __forwards_success
{
  _P __p_;

  template <class _R, ::cuda::std::enable_if_t<__has_on_success_with<_P, _R>, int> = 0>
  decltype(auto) on_success(_R&& __r)
  {
    return __p_.on_success(::cuda::std::forward<_R>(__r));
  }

  template <class _Self = _P, ::cuda::std::enable_if_t<__has_on_success_void<_Self>, int> = 0>
  decltype(auto) on_success()
  {
    return __p_.on_success();
  }

  template <class _Self = _P, ::cuda::std::enable_if_t<__has_on_enter<_Self>, int> = 0>
  void on_enter() noexcept(__on_enter_nothrow_v<_P>)
  {
    __p_.on_enter();
  }
};

// `_A` claims `_B` when a `catch (const _A&)` clause would take a thrown `_B`: same type, or
// publicly derived. is_same covers non-class types (is_base_of_v<int, int> is false).
template <class _A, class _B>
inline constexpr bool __claims = ::cuda::std::is_same_v<_A, _B> || ::cuda::std::is_base_of_v<_A, _B>;

template <class _B, class... _As>
inline constexpr bool __claimed_by_any = (__claims<_As, _B> || ...);

// Intra-pack subsumption: reject when any listed type claims another (duplicates included).
// Message names the dead Derived entry.
template <class...>
inline constexpr bool __catch_only_pack_ok = true;

template <class _Head, class... _Tail>
inline constexpr bool __catch_only_pack_ok<_Head, _Tail...> =
  (!__claims<_Head, _Tail> && ...) && (!__claims<_Tail, _Head> && ...) && __catch_only_pack_ok<_Tail...>;

// `catch_only<E1, E2, ...>(p)`: run `p`'s exception path when the active exception matches ANY
// listed type by catch-clause rules (same or publicly derived), else decline by rethrowing.
// The listed types may be anything catchable, std::exception heritage or not; matching is by
// re-observation, since a pack cannot expand into sibling catch clauses. Native C++ has no
// multi-type catch clause; this adds expressivity the language lacks. Policy parameter leads
// so the exception-type pack trails.
template <class _P, class... _Es>
struct __catch_only_t : __forwards_success<_P>
{
  using __exception_sink_tag = void;

  // Does the active exception match any listed type? A recursive ladder of re-observations;
  // the binding must be named for the no-exceptions expansion of _CCCL_CATCH.
  template <class _E0, class... _Rest>
  static bool __matches_active(const ::std::exception* __e)
  {
    // Fast path: for a class target and a std-derived active exception, the funnel pointer
    // decides via dynamic_cast -- no rethrow. (dynamic_cast agrees with catch matching:
    // ambiguous or non-public bases yield null, and a catch clause would not match either.)
    if constexpr (::cuda::std::is_class_v<_E0>)
    {
      if (__e)
      {
        if (dynamic_cast<const _E0*>(__e))
        {
          return true;
        }
        if constexpr (sizeof...(_Rest) > 0)
        {
          return __matches_active<_Rest...>(__e);
        }
        else
        {
          return false;
        }
      }
    }
    // Slow path: a non-class target, or a non-std active exception -- re-observe.
    _CCCL_TRY
    {
      throw;
    }
    _CCCL_CATCH ([[maybe_unused]] const _E0& __match)
    {
      return true;
    }
    _CCCL_CATCH_ALL
    {
      if constexpr (sizeof...(_Rest) > 0)
      {
        return __matches_active<_Rest...>(__e);
      }
      else
      {
        return false;
      }
    }
  }

  template <class _Fn, class _Self = _P, ::cuda::std::enable_if_t<__has_exception_hook<_Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    if (__matches_active<_Es...>(__exception))
    {
      // A matching non-std exception still reaches `_P` as a null pointer, per the funnel.
      return this->__p_(__exception, __loc, __fn);
    }
    throw; // decline: no listed type claims the active exception
  }
};

// Intra-pack duplicates are dead for exact matching; cone relations are fine (Base and
// Derived may both be listed, each matching only its own dynamic type).
template <class...>
inline constexpr bool __catch_exactly_pack_ok = true;

template <class _Head, class... _Tail>
inline constexpr bool __catch_exactly_pack_ok<_Head, _Tail...> =
  (!::cuda::std::is_same_v<_Head, _Tail> && ...) && __catch_exactly_pack_ok<_Tail...>;

// Is `_B` textually one of `_As...`? The exact-guard analogue of `__claimed_by_any`.
template <class _B, class... _As>
inline constexpr bool __listed_exactly = (::cuda::std::is_same_v<_As, _B> || ...);

// `catch_exactly<E1, E2, ...>(p)`: run `p`'s exception path when the active exception's
// DYNAMIC type is exactly one of the listed types, else decline by rethrowing. Monomorphic
// where `catch_only` is polymorphic: derived types do not match, so a handler accepts a type
// without inheriting its cone. Matching reads typeid through the std::exception funnel, so
// listed types must derive std::exception (enforced by the factory); a non-std active
// exception (null funnel) always declines.
template <class _P, class... _Es>
struct __catch_exactly_t : __forwards_success<_P>
{
  using __exception_sink_tag = void;

  template <class _Fn, class _Self = _P, ::cuda::std::enable_if_t<__has_exception_hook<_Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    if (__exception != nullptr && ((typeid(*__exception) == typeid(_Es)) || ...))
    {
      return this->__p_(__exception, __loc, __fn);
    }
    throw; // decline: the active exception's dynamic type is not listed
  }
};

// Tags a raw capability-bearing callable so normal forms are uniformly sink-typed.
// Forwards every capability it wraps; adds none. Storage follows the `subst_t<_R>`
// convention: an lvalue reaction is held by reference, an rvalue moved in.
template <class _P>
struct __as_policy : __forwards_success<_P>
{
  using __exception_sink_tag = void;

  template <class _Fn, class _Self = _P, ::cuda::std::enable_if_t<__has_exception_hook<_Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    return this->__p_(__exception, __loc, __fn);
  }
};

// Normalize any reaction into a policy object, returned by value (first match wins). A
// value stored inside carries its own ref-ness (e.g. subst_t<int&> for a reference result).
// Every normal form is sink-tagged: raw capability-bearing callables wrap in `__as_policy`.
template <class _R>
auto __normalize(_R&& __r)
{
  using _P = ::cuda::std::remove_cvref_t<_R>;
  if constexpr (__is_exception_sink_v<_P>)
  {
    // Already sink-tagged (named policies, composites, adapters, noop, ...).
    return static_cast<_P>(::cuda::std::forward<_R>(__r));
  }
  else if constexpr (__has_any_capability<_P>)
  {
    // Raw callable with a capability: tag it so identity elimination and &/| stay ADL-findable.
    return __as_policy<_R>{::cuda::std::forward<_R>(__r)};
  }
  else if constexpr (__is_ignore_v<_P>)
  {
    return __ignore_policy{};
  }
  else
  {
    return subst_t<_R>{::cuda::std::forward<_R>(__r)};
  }
}

// The type `__normalize` would produce for `_R`. After `__as_policy`, every normal form is
// sink-tagged, so identity elimination fires for every operand (ADDENDUM-1 §A.5 superseded).
template <class _R>
using __normalized_t = decltype(__normalize(::cuda::std::declval<_R>()));

template <class _R>
inline constexpr bool __normalizes_to_exception_sink_v = __is_exception_sink_v<__normalized_t<_R>>;

// Success-hook forwarding shared by the `&` and `|` composites: rightmost-wins. The outer
// `operator<<` enforces that the selected hook preserves the callable's expression type. The
// exception hook -- where the two composites differ -- lives in the derived types.
template <class _L, class _R>
struct __composite_hooks
{
  _L __l_;
  _R __r_;

  template <class _Rr,
            ::cuda::std::enable_if_t<__has_on_success_with<_R, _Rr> || __has_on_success_with<_L, _Rr>, int> = 0>
  decltype(auto) on_success(_Rr&& __r)
  {
    if constexpr (__has_on_success_with<_R, _Rr>)
    {
      return __r_.on_success(::cuda::std::forward<_Rr>(__r));
    }
    else
    {
      return __l_.on_success(::cuda::std::forward<_Rr>(__r));
    }
  }

  template <class _LL                                                                               = _L,
            class _RR                                                                               = _R,
            ::cuda::std::enable_if_t<__has_on_success_void<_RR> || __has_on_success_void<_LL>, int> = 0>
  decltype(auto) on_success()
  {
    if constexpr (__has_on_success_void<_R>)
    {
      return __r_.on_success();
    }
    else
    {
      return __l_.on_success();
    }
  }

  // Entry gates run left to right: each side may refuse the attempt before it starts.
  template <class _LL = _L, class _RR = _R, ::cuda::std::enable_if_t<__has_on_enter<_LL> || __has_on_enter<_RR>, int> = 0>
  void on_enter() noexcept(__on_enter_nothrow_v<_L> && __on_enter_nothrow_v<_R>)
  {
    if constexpr (__has_on_enter<_L>)
    {
      __l_.on_enter();
    }
    if constexpr (__has_on_enter<_R>)
    {
      __r_.on_enter();
    }
  }
};

// The sequencing composite `_L & _R`: on the exception path run `_L` then `_R`; `_R` answers.
// `&` discards non-final answers -- a non-final `retry` re-runs and discards, legal and almost
// never what you want.
template <class _L, class _R>
struct __policy_and : __composite_hooks<_L, _R>
{
  using __exception_sink_tag = void;

  static_assert(!__answers_nothing<_L>, "policies after a never-returning policy are unreachable");

  // Present iff either side has a hook. `_L` fires (answer discarded), then `_R` answers; with
  // no `_R` hook the composite's answer is `void`, which `__interpret_answer` rejects in final
  // position -- correct, since such a chain cannot answer on its own.
  template <class _Fn,
            class _LL                                                                             = _L,
            class _RR                                                                             = _R,
            ::cuda::std::enable_if_t<__has_exception_hook<_LL> || __has_exception_hook<_RR>, int> = 0>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    if constexpr (__has_exception_hook<_L>)
    {
      static_cast<void>(this->__l_(__exception, __loc, __fn)); // non-final answers are discarded
    }
    if constexpr (__has_exception_hook<_R>)
    {
      return this->__r_(__exception, __loc, __fn);
    }
  }
};

template <class _L, class _R>
__policy_and(_L, _R) -> __policy_and<_L, _R>;

// Forward declaration: `|` and `*` reuse this for arm answer interpretation (defined below).
template <class _Expr, class _P, class _Fn>
_Expr __interpret_answer(
  _P& __policy, const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn);

// The left arm of `|` provably starves the right when both are catch_only wrappers, the left's
// guard list claims every type the right lists, and the left's inner policy never declines a
// claimed exception. Sound and incomplete, like every dead-code theorem here: nested
// composites and raw guards escape the pattern; an inner policy that can decline (nothrow
// test fails) keeps the right arm live.
template <class _L, class _R>
inline constexpr bool __right_arm_starved = false;

template <class _P1, class... _As, class _P2, class... _Bs>
inline constexpr bool __right_arm_starved<__catch_only_t<_P1, _As...>, __catch_only_t<_P2, _Bs...>> =
  __exception_path_nothrow_v<_P1> && (__claimed_by_any<_Bs, _As...> && ...);

// A cone on the left starves an exact entry inside it on the right; an exact entry on the
// left starves only its own repetitions. The converse (exact left, cone right) never starves:
// the cone always has more members.
template <class _P1, class... _As, class _P2, class... _Bs>
inline constexpr bool __right_arm_starved<__catch_only_t<_P1, _As...>, __catch_exactly_t<_P2, _Bs...>> =
  __exception_path_nothrow_v<_P1> && (__claimed_by_any<_Bs, _As...> && ...);

template <class _P1, class... _As, class _P2, class... _Bs>
inline constexpr bool __right_arm_starved<__catch_exactly_t<_P1, _As...>, __catch_exactly_t<_P2, _Bs...>> =
  __exception_path_nothrow_v<_P1> && (__listed_exactly<_Bs, _As...> && ...);

// The alternation composite `_L | _R`: `_L` claims first; if it declines by throwing, `_R`
// handles the original (re-observed) exception. Each arm is called at the uniform 3-arg shape;
// acceptance is interpreted at `decltype(fn())`.
template <class _L, class _R>
struct __policy_or : __composite_hooks<_L, _R>
{
  using __exception_sink_tag = void;

  static_assert(__has_exception_hook<_L> && __has_exception_hook<_R>,
                "both sides of | must answer the exception path (have an exception hook)");
  static_assert(!__exception_path_nothrow_v<_L>,
                "the left policy never declines; alternatives after it are unreachable");
  static_assert(!__right_arm_starved<_L, _R>,
                "the left type guard already claims every exception type the right arm lists; "
                "the right alternative is unreachable");

  template <class _Fn,
            class _LL                                                                             = _L,
            class _RR                                                                             = _R,
            ::cuda::std::enable_if_t<__has_exception_hook<_LL> && __has_exception_hook<_RR>, int> = 0>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    using _Raw = decltype(__fn());

    const auto __right = [&](const ::std::exception* __cur) -> _Raw {
      return __interpret_answer<_Raw>(this->__r_, __cur, __loc, __fn);
    };
    const auto __reobserve_right = [&]() -> _Raw {
      _CCCL_TRY
      {
        throw;
      }
      _CCCL_CATCH (const ::std::exception& __e)
      {
        return __right(&__e);
      }
      _CCCL_CATCH_ALL
      {
        return __right(nullptr);
      }
    };

    // Interpret both arms at the callable's result type. Void callables surface ignore so
    // this composite can still sit as a top-level policy.
    _CCCL_TRY
    {
      if constexpr (::cuda::std::is_void_v<_Raw>)
      {
        __interpret_answer<_Raw>(this->__l_, __exception, __loc, __fn);
        return ::std::ignore;
      }
      else
      {
        return __interpret_answer<_Raw>(this->__l_, __exception, __loc, __fn);
      }
    }
    _CCCL_CATCH_ALL
    {
      if constexpr (::cuda::std::is_void_v<_Raw>)
      {
        __reobserve_right();
        return ::std::ignore;
      }
      else
      {
        return __reobserve_right();
      }
    }
  }
};

template <class _L, class _R>
__policy_or(_L, _R) -> __policy_or<_L, _R>;

// `p * n`: behaviorally the n-fold `|` of p with itself. One stored policy, invoked up to n
// times; the active exception is re-observed between iterations exactly as `__policy_or` does
// between arms. `n == 0` declines immediately (the empty fold is rethrow). The stored policy's
// hook is invoked up to n times; with the inventory now stateless this needs no copying --
// user-defined policies should likewise tolerate re-invocation.
template <class _P>
struct __policy_pow : __forwards_success<_P>
{
  using __exception_sink_tag = void;
  int __n_;

  static_assert(__has_exception_hook<_P>,
                "the repeated policy must answer the exception path (have an exception hook)");
  static_assert(!__exception_path_nothrow_v<_P>,
                "the repeated policy never declines; repetitions after the first are unreachable");

  template <class _Fn>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    using _Expr = decltype(__fn());
    if (__n_ == 0)
    {
      throw; // empty fold: decline with the still-active exception
    }

    // Recurse inside the catch so the re-observed exception pointer stays alive for the
    // next arm (same lifetime rule as `__policy_or`). The recursion is bounded: `__left`
    // decreases every level and `__left == 1` declines by rethrowing. gcc 14.3+/15's
    // -Winfinite-recursion is blind to exceptional exits and misreads instantiations whose
    // only normal returns are the recursive calls (e.g. a never-returning repeated policy).
    _CCCL_DIAG_PUSH
    _CCCL_DIAG_SUPPRESS_GCC("-Wpragmas") // gcc < 12 does not know the warning below; without this
                                         // line the unknown name itself trips -Werror=pragmas
    _CCCL_DIAG_SUPPRESS_GCC("-Winfinite-recursion")
    const auto __go = [&](auto& __self, const ::std::exception* __cur, int __left) -> _Expr {
      _CCCL_TRY
      {
        return __interpret_answer<_Expr>(this->__p_, __cur, __loc, __fn);
      }
      _CCCL_CATCH_ALL
      {
        if (__left == 1)
        {
          throw;
        }
        _CCCL_TRY
        {
          throw;
        }
        _CCCL_CATCH (const ::std::exception& __e)
        {
          return __self(__self, &__e, __left - 1);
        }
        _CCCL_CATCH_ALL
        {
          return __self(__self, nullptr, __left - 1);
        }
      }
    };
    if constexpr (::cuda::std::is_void_v<_Expr>)
    {
      __go(__go, __exception, __n_);
      return ::std::ignore;
    }
    else
    {
      return __go(__go, __exception, __n_);
    }
    _CCCL_DIAG_POP
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

template <class _T, bool = ::cuda::std::is_enum_v<_T>>
struct __integral_base
{
  using type = _T;
};
template <class _T>
struct __integral_base<_T, true>
{
  using type = ::cuda::std::underlying_type_t<_T>;
};

template <class _From, class _To>
constexpr bool __value_preserving_impl()
{
  using _F = typename __integral_base<::cuda::std::remove_cvref_t<_From>>::type;
  using _T = ::cuda::std::remove_cvref_t<_To>;
  if constexpr (!::cuda::std::is_arithmetic_v<_F> || !::cuda::std::is_arithmetic_v<_T>)
  {
    return true; // non-arithmetic pairs: the is_convertible baseline is the whole law
  }
  else if constexpr (::cuda::std::is_floating_point_v<_T>)
  {
    return true; // precision loss is tolerated where range loss is not
  }
  else if constexpr (::cuda::std::is_floating_point_v<_F>)
  {
    return false; // floating never converts to integral
  }
  else
  {
    // Integral range containment, sign-aware. A signed source holds negatives an unsigned
    // destination cannot; equal signedness compares widths; unsigned-to-signed needs strictly
    // more width to cover the source's maximum.
    constexpr bool __f_signed = ::cuda::std::is_signed_v<_F>;
    constexpr bool __t_signed = ::cuda::std::is_signed_v<_T>;
    if constexpr (__f_signed && !__t_signed)
    {
      return false;
    }
    else if constexpr (__f_signed == __t_signed)
    {
      return sizeof(_F) <= sizeof(_T);
    }
    else
    {
      return sizeof(_F) < sizeof(_T);
    }
  }
}

template <class _From, class _To>
inline constexpr bool __value_preserving_v =
  ::cuda::std::is_convertible_v<_From, _To> && __value_preserving_impl<_From, _To>();

// Interpret the final element's answer as the expression's value, converting to `_Expr`.
template <class _Expr, class _P, class _Fn>
_Expr __interpret_answer(
  _P& __policy, const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
{
  using _Answer = __hook_answer_t<_P, _Fn>;
  static_assert(!::cuda::std::is_void_v<_Answer>,
                "the final policy must answer the exception path: nullval to die, ::std::ignore "
                "to resume, or a value to substitute");

  if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Answer>, nullval>)
  {
    // Never returns: no backstop beyond the unreachable marker.
    __policy(__exception, __loc, __fn);
    _CCCL_UNREACHABLE();
  }
  else if constexpr (__is_ignore_v<_Answer>)
  {
    // Resume: default-construct the expression's value (nothing to do for void).
    static_cast<void>(__policy(__exception, __loc, __fn));
    if constexpr (!::cuda::std::is_void_v<_Expr>)
    {
      static_assert(!::cuda::std::is_reference_v<_Expr>,
                    "an on_throw reaction that resumes has nothing to refer to for a reference result");
      static_assert(::cuda::std::is_default_constructible_v<_Expr>,
                    "an on_throw reaction that resumes requires a default-constructible result");
      return _Expr{};
    }
  }
  else
  {
    // Substitute: convert the answer to the expression's type. A reference result may be served
    // only by an lvalue answer of a compatible type; anything else dies with the call.
    static_assert(
      !::cuda::std::is_reference_v<_Expr>
        || (::cuda::std::is_lvalue_reference_v<_Answer> && ::cuda::std::is_lvalue_reference_v<_Expr>
            && ::cuda::std::is_convertible_v<::cuda::std::remove_reference_t<_Answer>*,
                                             ::cuda::std::remove_reference_t<_Expr>*>),
      "a reference result needs an on_throw reaction passed as an lvalue of the same "
      "type, anything else dying with the call");
    static_assert(::cuda::std::is_convertible_v<_Answer, _Expr>,
                  "an on_throw reaction is a policy, a never-returning callable (one returning "
                  "nullval, like abort and terminate), ::std::ignore, or a value convertible to "
                  "the result of the callable");
    static_assert(__value_preserving_v<_Answer, _Expr>,
                  "the policy's answer does not preserve the callable's value range (for example "
                  "an int answer under an unsigned result); write the conversion in the policy -- "
                  "subst(0xffffffffu), not subst(-1) -- if the narrowing is intended");
    return static_cast<_Expr>(__policy(__exception, __loc, __fn));
  }
}

// Walk the chain on the exception path: with no answering hook anywhere, propagate; else
// interpret. The parameters go unread in the propagate instantiation, which gcc 9 flags
// without the attribute.
template <class _Expr, class _P, class _Fn>
_Expr __on_exception(_P& __policy,
                     [[maybe_unused]] const ::std::exception* __exception,
                     [[maybe_unused]] const ::cuda::std::source_location __loc,
                     [[maybe_unused]] _Fn& __fn)
{
  if constexpr (!__has_exception_hook<_P, _Fn>)
  {
    throw; // no element answered: let the exception propagate
  }
  else
  {
    return __interpret_answer<_Expr>(__policy, __exception, __loc, __fn);
  }
}

// The policy carrier. ADL finds `operator<<` here since the type lives in this namespace.
template <class _Reaction>
struct __on_throw_policy
{
  _Reaction __reaction_;
  const ::cuda::std::source_location __loc_;
};

template <class _R>
__on_throw_policy(_R, ::cuda::std::source_location) -> __on_throw_policy<_R>;

template <class _Reaction, class _Fn>
// A resuming chain reads neither exception nor location in some instantiations; gcc 9 flags the
// unread policy without the attribute.
decltype(auto) operator<<([[maybe_unused]] __on_throw_policy<_Reaction> __policy, _Fn&& __fn) noexcept(
  __exception_path_nothrow_v<_Reaction, _Fn> && __on_enter_nothrow_v<_Reaction>)
{
  // Bind as a non-const lvalue: a hook may invoke it again later.
  _Fn& __f = __fn;

  // A `noexcept` callable puts the policy out of reach: an exception raised inside it ends the
  // program where it stands, so the catch below could never run and the policy would be a
  // promise nobody keeps.
  static_assert(!noexcept(__f()),
                "on_throw has nothing to do for a noexcept callable, which terminates rather than "
                "throws; call such a callable directly");

  using _Expr = decltype(__f());
  using _P    = _Reaction;

  // The entry gate runs before the attempt and outside the policy's own catch: an exception
  // thrown here (a gate refusing the attempt) belongs to the enclosing scope, never to the
  // policy that raised it.
  if constexpr (__has_on_enter<_P>)
  {
    static_assert(::cuda::std::is_void_v<__on_enter_of<_P>>,
                  "on_enter answers nothing: it admits the attempt or refuses it by throwing");
    __policy.__reaction_.on_enter();
  }

  if constexpr (::cuda::std::is_void_v<_Expr>)
  {
    _CCCL_TRY
    {
      __f();
      if constexpr (__has_on_success_void<_P>)
      {
        using _Answer = decltype(::cuda::std::declval<_P&>().on_success());
        static_assert(::cuda::std::is_same_v<_Answer, _Expr>,
                      "a policy's on_success must preserve the expression type; policies no "
                      "longer own it (SPEC-ADDENDUM-7)");
        __policy.__reaction_.on_success();
      }
    }
    _CCCL_CATCH (const ::std::exception& __exception)
    {
      return detail::__on_exception<_Expr>(__policy.__reaction_, &__exception, __policy.__loc_, __f);
    }
    _CCCL_CATCH_ALL
    {
      return detail::__on_exception<_Expr>(__policy.__reaction_, nullptr, __policy.__loc_, __f);
    }
  }
  else
  {
    _CCCL_TRY
    {
      if constexpr (__has_on_success_with<_P, _Expr>)
      {
        using _Answer = decltype(::cuda::std::declval<_P&>().on_success(::cuda::std::declval<_Expr>()));
        static_assert(::cuda::std::is_same_v<_Answer, _Expr>,
                      "a policy's on_success must preserve the expression type; policies no "
                      "longer own it (SPEC-ADDENDUM-7)");
        return __policy.__reaction_.on_success(__f());
      }
      else
      {
        return __f();
      }
    }
    _CCCL_CATCH (const ::std::exception& __exception)
    {
      return detail::__on_exception<_Expr>(__policy.__reaction_, &__exception, __policy.__loc_, __f);
    }
    _CCCL_CATCH_ALL
    {
      return detail::__on_exception<_Expr>(__policy.__reaction_, nullptr, __policy.__loc_, __f);
    }
  }
}
} // namespace detail
#endif // !_CCCL_DOXYGEN_INVOKED

/**
 * @brief Restricts a policy to exceptions matching any of `E1, E2, ...`: `catch_only<E...>(p)`
 * runs `p`'s exception path when the active exception matches any listed type by catch-clause
 * rules (same or publicly derived), and otherwise declines by rethrowing. The listed types may
 * be anything catchable -- std::exception derivatives, user structs, even `int`. Native C++
 * has no multi-type catch clause; this adds that expressivity. A matching exception that does
 * not derive from `std::exception` reaches `p`'s hook as a null pointer. A pack where one type
 * claims another (identical or base-of) is rejected -- the claimed entry would be dead.
 */
template <class... _Es, class _P>
auto catch_only(_P&& __p)
{
  static_assert(sizeof...(_Es) > 0, "catch_only requires at least one exception type");
  static_assert(detail::__catch_only_pack_ok<_Es...>,
                "catch_only<..., Base, ..., Derived, ...>: the Derived entry is dead "
                "(Base already claims it)");
  auto __np = detail::__normalize(::cuda::std::forward<_P>(__p));
  return detail::__catch_only_t<decltype(__np), _Es...>{::cuda::std::move(__np)};
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
template <class... _Es, class _P>
auto catch_exactly(_P&& __p)
{
  static_assert(sizeof...(_Es) > 0, "catch_exactly requires at least one exception type");
  static_assert((::cuda::std::is_base_of_v<::std::exception, _Es> && ...),
                "catch_exactly matches dynamic types through the std::exception funnel; every "
                "listed type must derive std::exception (catch_only takes anything catchable)");
  static_assert(detail::__catch_exactly_pack_ok<_Es...>,
                "catch_exactly<..., E, ..., E, ...>: a repeated entry is dead");
  auto __np = detail::__normalize(::cuda::std::forward<_P>(__p));
  return detail::__catch_exactly_t<decltype(__np), _Es...>{::cuda::std::move(__np)};
}

/**
 * @brief Sequences two policies: on the exception path `_L` runs then `_R`, and `_R`'s answer
 * decides. `noop` is the two-sided identity. Constrained so at least one operand is a policy
 * this header defines, so it never hijacks unrelated `&` expressions; a chain of plain lambdas
 * is therefore not composable, but heading it with `noop` makes it so.
 */
template <class _L,
          class _R,
          ::cuda::std::enable_if_t<detail::__is_exception_sink_v<_L> || detail::__is_exception_sink_v<_R>, int> = 0>
auto operator&(_L&& __l, _R&& __r)
{
  return detail::__policy_and{
    detail::__normalize(::cuda::std::forward<_L>(__l)), detail::__normalize(::cuda::std::forward<_R>(__r))};
}

//! @brief Left identity of `&`: `noop & p` is `__normalize(p)` when that result is sink-tagged.
template <class _R, ::cuda::std::enable_if_t<detail::__normalizes_to_exception_sink_v<_R>, int> = 0>
auto operator&(noop_t, _R&& __r)
{
  return detail::__normalize(::cuda::std::forward<_R>(__r));
}

//! @brief Right identity of `&`. `noop` itself is excluded so `noop & noop` is not ambiguous.
template <class _L,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_L>, noop_t>
                                     && detail::__normalizes_to_exception_sink_v<_L>,
                                   int> = 0>
auto operator&(_L&& __l, noop_t)
{
  return detail::__normalize(::cuda::std::forward<_L>(__l));
}

/**
 * @brief Alternation: `_L` gets first claim; if it declines by throwing, `_R` handles the
 * original exception. `rethrow` is the two-sided identity. Same operand constraint as `&`.
 */
template <class _L,
          class _R,
          ::cuda::std::enable_if_t<detail::__is_exception_sink_v<_L> || detail::__is_exception_sink_v<_R>, int> = 0>
auto operator|(_L&& __l, _R&& __r)
{
  return detail::__policy_or{
    detail::__normalize(::cuda::std::forward<_L>(__l)), detail::__normalize(::cuda::std::forward<_R>(__r))};
}

//! @brief Left identity of `|`: `rethrow | p` is `__normalize(p)` when that result is sink-tagged.
template <class _R, ::cuda::std::enable_if_t<detail::__normalizes_to_exception_sink_v<_R>, int> = 0>
auto operator|(rethrow_t, _R&& __r)
{
  return detail::__normalize(::cuda::std::forward<_R>(__r));
}

//! @brief Right identity of `|`. `rethrow` itself is excluded so `rethrow | rethrow` is not ambiguous.
template <class _L,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_L>, rethrow_t>
                                     && detail::__normalizes_to_exception_sink_v<_L>,
                                   int> = 0>
auto operator|(_L&& __l, rethrow_t)
{
  return detail::__normalize(::cuda::std::forward<_L>(__l));
}

/**
 * @brief Repetition: `p * n` is the n-fold `|` of `p` with itself -- behaviorally
 * `p | p | ... | p` (n copies). Laws: `p * 0` ≡ `rethrow` (empty fold); `p * 1` ≡ `p`
 * (behaviorally); `p * (m + n)` ≡ `p * m | p * n`. `*` binds tighter than `&` and `|`, so
 * `(notify & retry) * 3` notifies before each re-attempt, while `notify & retry * 3`
 * notifies once then re-attempts three times.
 */
template <class _P,
          ::cuda::std::enable_if_t<detail::__is_exception_sink_v<::cuda::std::remove_cvref_t<_P>>
                                     || detail::__has_any_capability<::cuda::std::remove_cvref_t<_P>>,
                                   int> = 0>
auto operator*(_P&& __p, int __n)
{
  _CCCL_ASSERT(__n >= 0, "repetition requires a non-negative count");
  using _Np = detail::__normalized_t<_P>;
  return detail::__policy_pow<_Np>{{detail::__normalize(::cuda::std::forward<_P>(__p))}, __n};
}

//! @brief Commuted form of `operator*`: `n * p` is `p * n`.
template <class _P,
          ::cuda::std::enable_if_t<detail::__is_exception_sink_v<::cuda::std::remove_cvref_t<_P>>
                                     || detail::__has_any_capability<::cuda::std::remove_cvref_t<_P>>,
                                   int> = 0>
auto operator*(int __n, _P&& __p)
{
  return ::cuda::std::forward<_P>(__p) * __n;
}

/**
 * @brief Restricts a policy with a runtime predicate.
 *
 * Returns `when_t{pred} & p`: as a `|` arm, false means the next alternative gets the
 * exception; inside a larger `&`, false declines the whole sequence.
 */
template <class _Pred, class _P>
auto when(_Pred&& __pred, _P&& __p)
{
  return when_t<_Pred>{::cuda::std::forward<_Pred>(__pred)} & detail::__normalize(::cuda::std::forward<_P>(__p));
}

/**
 * @brief Runs the head, then every finalizer, on the exception path -- the finalizers run
 * whether the head accepts or declines. The composite's answer is the HEAD's; finalizers'
 * answers are discarded. If the head declines, the finalizers observe the in-flight
 * exception and the head's exception then continues onward (a finalizer's own throw
 * replaces it, as anywhere in C++). Later arguments are increasingly unconditional.
 * A named function, not an operator: it has no left identity.
 */
template <class _A, class _B>
struct always_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond
  _A __head_;
  _B __fin_;

  template <class _Fn>
  decltype(auto) operator()(const ::std::exception* __e, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    _CCCL_TRY
    {
      using _Ans = detail::__hook_answer_t<_A, _Fn>;
      if constexpr (::cuda::std::is_void_v<_Ans>)
      {
        __head_(__e, __loc, __fn);
        __fin_(__e, __loc, __fn);
        return;
      }
      else if constexpr (detail::__answers_nothing<_A, _Fn>)
      {
        // Never actually returns (the decline path below runs the finalizer); returning the
        // call keeps the composite's answer type `nullval` via guaranteed elision.
        return __head_(__e, __loc, __fn);
      }
      else
      {
        decltype(auto) __r = __head_(__e, __loc, __fn);
        static_cast<void>(__fin_(__e, __loc, __fn)); // a finalizer's answer is discarded
        return static_cast<_Ans>(__r);
      }
    }
    _CCCL_CATCH_ALL
    {
      // The head declined (or a finalizer threw after an accept): run the finalizer with the
      // CURRENT exception, then let that exception continue onward.
      _CCCL_TRY
      {
        throw;
      }
      _CCCL_CATCH (const ::std::exception& __cur)
      {
        static_cast<void>(__fin_(&__cur, __loc, __fn));
      }
      _CCCL_CATCH_ALL
      {
        static_cast<void>(__fin_(nullptr, __loc, __fn));
      }
      throw;
    }
  }

  // The head's success hook is a type-preserving side-effect; forward it.
  template <class _R, ::cuda::std::enable_if_t<detail::__has_on_success_with<_A, _R>, int> = 0>
  decltype(auto) on_success(_R&& __r)
  {
    return __head_.on_success(::cuda::std::forward<_R>(__r));
  }

  template <class _Self = _A, ::cuda::std::enable_if_t<detail::__has_on_success_void<_Self>, int> = 0>
  decltype(auto) on_success()
  {
    return __head_.on_success();
  }

  // Entry gates forward to head and finalizer alike: each may refuse the attempt.
  template <class _AA                                                                                 = _A,
            class _BB                                                                                 = _B,
            ::cuda::std::enable_if_t<detail::__has_on_enter<_AA> || detail::__has_on_enter<_BB>, int> = 0>
  void on_enter() noexcept(detail::__on_enter_nothrow_v<_A> && detail::__on_enter_nothrow_v<_B>)
  {
    if constexpr (detail::__has_on_enter<_A>)
    {
      __head_.on_enter();
    }
    if constexpr (detail::__has_on_enter<_B>)
    {
      __fin_.on_enter();
    }
  }
};

//! @brief See @ref always_t. Variadic: `always(a, b, c)` folds left, so both `b` and `c` run
//! regardless of `a`, and `c` runs regardless of `b`.
template <class _A, class _B, class... _Cs>
auto always(_A&& __a, _B&& __b, _Cs&&... __cs)
{
  if constexpr (sizeof...(_Cs) == 0)
  {
    auto __ha = detail::__normalize(::cuda::std::forward<_A>(__a));
    auto __hb = detail::__normalize(::cuda::std::forward<_B>(__b));
    return always_t<decltype(__ha), decltype(__hb)>{::cuda::std::move(__ha), ::cuda::std::move(__hb)};
  }
  else
  {
    return always(always(::cuda::std::forward<_A>(__a), ::cuda::std::forward<_B>(__b)),
                  ::cuda::std::forward<_Cs>(__cs)...);
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

    sink_base(answer_kind __kind                    = answer_kind::passthrough,
              bool __may_rethrow                    = true,
              long long __min_value                 = 0,
              unsigned long long __max_value        = 0,
              const ::std::type_info* __answer_type = nullptr,
              ::std::string_view __answer_name      = {})
        : kind(__kind)
        , may_rethrow(__may_rethrow)
        , min_value(__min_value)
        , max_value(__max_value)
        , answer_type(__answer_type)
        , answer_name(__answer_name)
    {}
    virtual ~sink_base()             = default;
    virtual sink_base* clone() const = 0;
    virtual ::std::any hook(const ::std::exception*, ::cuda::std::source_location, ::std::function<::std::any()>&) = 0;
    virtual ::std::any on_success(::std::any __boxed)
    {
      return __boxed;
    }
    virtual ::std::any on_success()
    {
      return {};
    }

    //! @brief Entry gate, run before each attempt. Default: admit.
    virtual void on_enter() {}
  };

private:
  template <class _Ans>
  static constexpr answer_kind __kind_of()
  {
    using _T = ::cuda::std::remove_cvref_t<_Ans>;
    if constexpr (::cuda::std::is_void_v<_T>)
    {
      return answer_kind::effects;
    }
    else if constexpr (detail::__is_ignore_v<_T>)
    {
      return answer_kind::resumes;
    }
    else if constexpr (::cuda::std::is_same_v<_T, nullval>)
    {
      return answer_kind::dies;
    }
    else if constexpr (::cuda::std::is_same_v<_T, ::std::any>)
    {
      return answer_kind::passthrough;
    }
    else if constexpr (::cuda::std::is_enum_v<_T> || ::cuda::std::is_integral_v<_T>)
    {
      return answer_kind::integral;
    }
    else if constexpr (::cuda::std::is_floating_point_v<_T>)
    {
      return answer_kind::floating;
    }
    else
    {
      return answer_kind::udt;
    }
  }

  template <class _Int>
  static constexpr long long __limits_min()
  {
    if constexpr (::cuda::std::is_unsigned_v<_Int>)
    {
      return 0;
    }
    else
    {
      return static_cast<long long>(::std::numeric_limits<_Int>::min());
    }
  }

  template <class _Int>
  static constexpr unsigned long long __limits_max()
  {
    return static_cast<unsigned long long>(::std::numeric_limits<_Int>::max());
  }

  template <class _Ans>
  static constexpr long long __stored_min()
  {
    using _T = ::cuda::std::remove_cvref_t<_Ans>;
    if constexpr (__kind_of<_T>() == answer_kind::integral)
    {
      if constexpr (::cuda::std::is_enum_v<_T>)
      {
        return __limits_min<::cuda::std::underlying_type_t<_T>>();
      }
      else
      {
        return __limits_min<_T>();
      }
    }
    else
    {
      return 0;
    }
  }

  template <class _Ans>
  static constexpr unsigned long long __stored_max()
  {
    using _T = ::cuda::std::remove_cvref_t<_Ans>;
    if constexpr (__kind_of<_T>() == answer_kind::integral)
    {
      if constexpr (::cuda::std::is_enum_v<_T>)
      {
        return __limits_max<::cuda::std::underlying_type_t<_T>>();
      }
      else
      {
        return __limits_max<_T>();
      }
    }
    else
    {
      return 0;
    }
  }

  //! The one universal erasure path: instantiate the wrapped policy's
  //! templated hook at the boxing proxy; derive all metadata statically.
  template <class _P>
  struct __model final : sink_base
  {
    _P __p_;

    using __ans_t                        = detail::__hook_answer_t<_P, ::std::function<::std::any()>>;
    static constexpr answer_kind __skind = __kind_of<__ans_t>();

    explicit __model(_P __p)
        : sink_base(
            __skind,
            !detail::__exception_path_nothrow_v<_P, ::std::function<::std::any()>>,
            __stored_min<__ans_t>(),
            __stored_max<__ans_t>(),
            (__skind == answer_kind::passthrough || __skind == answer_kind::dies || __skind == answer_kind::resumes
             || __skind == answer_kind::effects)
              ? nullptr
              : &typeid(::cuda::std::remove_cvref_t<__ans_t>),
            type_name<::cuda::std::remove_cvref_t<__ans_t>>)
        , __p_(::cuda::std::move(__p))
    {}

    sink_base* clone() const override
    {
      return new __model(*this); // a fresh policy copy: re-armed state
    }

    ::std::any
    hook(const ::std::exception* __e, ::cuda::std::source_location __loc, ::std::function<::std::any()>& __fn) override
    {
      if constexpr (__skind == answer_kind::effects || __skind == answer_kind::resumes)
      {
        static_cast<void>(__p_(__e, __loc, __fn));
        return {};
      }
      else if constexpr (__skind == answer_kind::dies)
      {
        static_cast<void>(__p_(__e, __loc, __fn));
        _CCCL_UNREACHABLE();
      }
      else if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<__ans_t>, ::std::any>)
      {
        return __p_(__e, __loc, __fn); // the value flowed through the callable; no re-box
      }
      else
      {
        return ::std::any(__p_(__e, __loc, __fn)); // a stored value, boxed as its own type
      }
    }

    ::std::any on_success(::std::any __boxed) override
    {
      if constexpr (detail::__has_on_success_with<_P, ::std::any>)
      {
        return ::std::any(__p_.on_success(::cuda::std::move(__boxed)));
      }
      else
      {
        // Identity. NOTE: a wrapped success channel that cannot accept
        // `std::any` (it needs the real type, like remember's store) lands
        // here too -- its success feature does not survive erasure.
        return __boxed;
      }
    }
    ::std::any on_success() override
    {
      if constexpr (detail::__has_on_success_void<_P>)
      {
        if constexpr (::cuda::std::is_void_v<decltype(__p_.on_success())>)
        {
          __p_.on_success();
          return {};
        }
        else
        {
          return ::std::any(__p_.on_success());
        }
      }
      else
      {
        return {};
      }
    }
    void on_enter() override
    {
      if constexpr (detail::__has_on_enter<_P>)
      {
        __p_.on_enter();
      }
    }
  };

  ::std::unique_ptr<sink_base> __p_;

  [[noreturn]] void __throw_mismatch(::std::string_view __stored, ::std::string_view __wanted) const
  {
    ::std::string __msg{"exception_sink cannot convert "};
    __msg.append(__stored.data(), __stored.size());
    __msg.append(" answer to ");
    __msg.append(__wanted.data(), __wanted.size());
    throw ::std::logic_error(__msg);
  }

  template <class _Int>
  [[nodiscard]] bool __range_contains_int() const
  {
    const auto __raw_max = ::std::numeric_limits<_Int>::max();
    if constexpr (::cuda::std::is_unsigned_v<_Int>)
    {
      if (__p_->min_value < 0)
      {
        return false;
      }
      return __p_->max_value <= static_cast<unsigned long long>(__raw_max);
    }
    else
    {
      const auto __raw_min = ::std::numeric_limits<_Int>::min();
      if (__p_->min_value < static_cast<long long>(__raw_min))
      {
        return false;
      }
      return __p_->max_value <= static_cast<unsigned long long>(__raw_max);
    }
  }

  template <class _Raw>
  [[nodiscard]] bool __range_contains() const
  {
    using _T = ::cuda::std::remove_cvref_t<_Raw>;
    if constexpr (::cuda::std::is_enum_v<_T>)
    {
      return __range_contains_int<::cuda::std::underlying_type_t<_T>>();
    }
    else
    {
      return __range_contains_int<_T>();
    }
  }

  template <class _Raw>
  void __check_compatible() const
  {
    using _T                             = ::cuda::std::remove_cvref_t<_Raw>;
    const answer_kind __kind             = __p_->kind;
    [[maybe_unused]] const auto __wanted = type_name<_T>;
    [[maybe_unused]] const auto __stored =
      __p_->answer_name.empty() ? ::std::string_view{"<erased>"} : __p_->answer_name;
    if (__kind == answer_kind::dies || __kind == answer_kind::resumes || __kind == answer_kind::effects
        || __kind == answer_kind::passthrough)
    {
      return;
    }
    if constexpr (::cuda::std::is_void_v<_Raw>)
    {
      __throw_mismatch(__stored, __wanted);
    }
    else if (__kind == answer_kind::integral)
    {
      if constexpr (::cuda::std::is_floating_point_v<_T>)
      {
        return;
      }
      else if constexpr (::cuda::std::is_integral_v<_T> || ::cuda::std::is_enum_v<_T>)
      {
        if (__range_contains<_Raw>())
        {
          return;
        }
      }
      __throw_mismatch(__stored, __wanted);
    }
    else if (__kind == answer_kind::floating)
    {
      if constexpr (::cuda::std::is_floating_point_v<_T>)
      {
        return;
      }
      __throw_mismatch(__stored, __wanted);
    }
    else if (__kind == answer_kind::udt)
    {
      if (__p_->answer_type && *__p_->answer_type == typeid(_T))
      {
        return;
      }
      __throw_mismatch(__stored, __wanted);
    }
  }

  template <class _Raw>
  [[nodiscard]] _Raw __unbox(const ::std::any& __box) const
  {
    using _T = ::cuda::std::remove_cvref_t<_Raw>;
    if constexpr (::cuda::std::is_same_v<_T, ::std::any>)
    {
      return __box;
    }
    else if (!__box.has_value())
    {
      if constexpr (::cuda::std::is_default_constructible_v<_T>)
      {
        return _T{};
      }
      else
      {
        __throw_mismatch("<empty>", type_name<_T>);
      }
    }
    else if (const _T* __exact = ::std::any_cast<_T>(&__box))
    {
      return *__exact;
    }
    else if constexpr (::cuda::std::is_arithmetic_v<_T> || ::cuda::std::is_enum_v<_T>)
    {
      _T __out{};
      bool __hit         = false;
      bool __found_lossy = false;
      // The same conversion law, per VALUE: the box is opaque to the first-use type check
      // (passthrough composites), so representability is decided on the number itself.
      const auto __fits = [](auto __v) -> bool {
        using _B = typename detail::__integral_base<_T>::type;
        if constexpr (::cuda::std::is_signed_v<decltype(__v)>)
        {
          if (__v < 0)
          {
            if constexpr (::cuda::std::is_signed_v<_B>)
            {
              return static_cast<long long>(__v) >= static_cast<long long>(::std::numeric_limits<_B>::min());
            }
            else
            {
              return false;
            }
          }
        }
        return static_cast<unsigned long long>(__v)
            <= static_cast<unsigned long long>(::std::numeric_limits<_B>::max());
      };
      const auto __accept = [&](auto __tag) {
        using _Stored = typename decltype(__tag)::type;
        if (__hit || __found_lossy)
        {
          return;
        }
        if (const _Stored* __p = ::std::any_cast<_Stored>(&__box))
        {
          if constexpr (::cuda::std::is_floating_point_v<typename detail::__integral_base<_T>::type>)
          {
            __out = static_cast<_T>(*__p); // anything -> floating: by fiat
            __hit = true;
          }
          else if constexpr (::cuda::std::is_floating_point_v<_Stored>)
          {
            __found_lossy = true; // floating never converts to integral
          }
          else if (__fits(*__p))
          {
            __out = static_cast<_T>(*__p);
            __hit = true;
          }
          else
          {
            __found_lossy = true; // right category, unrepresentable value
          }
        }
      };
      __accept(::cuda::std::type_identity<bool>{});
      __accept(::cuda::std::type_identity<char>{});
      __accept(::cuda::std::type_identity<signed char>{});
      __accept(::cuda::std::type_identity<unsigned char>{});
      __accept(::cuda::std::type_identity<short>{});
      __accept(::cuda::std::type_identity<unsigned short>{});
      __accept(::cuda::std::type_identity<int>{});
      __accept(::cuda::std::type_identity<unsigned>{});
      __accept(::cuda::std::type_identity<long>{});
      __accept(::cuda::std::type_identity<unsigned long>{});
      __accept(::cuda::std::type_identity<long long>{});
      __accept(::cuda::std::type_identity<unsigned long long>{});
      __accept(::cuda::std::type_identity<float>{});
      __accept(::cuda::std::type_identity<double>{});
      __accept(::cuda::std::type_identity<long double>{});
      if (!__hit)
      {
        __throw_mismatch(__box.type().name(), type_name<_T>);
      }
      return __out;
    }
    else
    {
      __throw_mismatch(__box.type().name(), type_name<_T>);
    }
  }

public:
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  //! @brief Erase a policy. Prefer the @ref type_erase factory, which also
  //! normalizes the historical reactions.
  template <class _P,
            ::cuda::std::enable_if_t<detail::__is_exception_sink_v<_P>
                                       && !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_P>, exception_sink>,
                                     int> = 0>
  explicit exception_sink(_P __p)
      : __p_(new __model<_P>(::cuda::std::move(__p)))
  {}

  //! @brief Adopt a custom model derived from @ref sink_base.
  explicit exception_sink(::std::unique_ptr<sink_base> __custom)
      : __p_(::std::move(__custom))
  {
    _CCCL_ASSERT(__p_, "exception_sink requires a non-null model");
  }

  exception_sink(const exception_sink& __other)
      : __p_(__other.__p_->clone())
  {}
  exception_sink(exception_sink&&) noexcept            = default;
  exception_sink& operator=(exception_sink&&) noexcept = default;
  exception_sink& operator=(const exception_sink& __other)
  {
    __p_.reset(__other.__p_->clone());
    return *this;
  }

  [[nodiscard]] answer_kind kind() const noexcept
  {
    return __p_->kind;
  }
  [[nodiscard]] bool may_rethrow() const noexcept
  {
    return __p_->may_rethrow;
  }

  //! @brief The uniform hook: answers `decltype(fn())`. A void callable answers
  //! `decltype(::std::ignore)` so the presence-probe archetype stays admissible.
  template <class _Fn, class _Raw = decltype(::cuda::std::declval<_Fn&>()())>
  auto operator()(const ::std::exception* __e, const ::cuda::std::source_location __loc, _Fn& __fn)
    -> ::cuda::std::conditional_t<::cuda::std::is_void_v<_Raw>, decltype(::std::ignore), _Raw>
  {
    static_assert(!::cuda::std::is_reference_v<_Raw>,
                  "exception_sink cannot serve a reference-returning callable: std::any cannot "
                  "carry references; use a concrete policy, or return a pointer");
    if constexpr (::cuda::std::is_reference_v<_Raw>)
    {
      _CCCL_UNREACHABLE();
    }
    else
    {
      ::std::function<::std::any()> __proxy = [&__fn]() -> ::std::any {
        if constexpr (::cuda::std::is_void_v<_Raw>)
        {
          __fn();
          return {};
        }
        else
        {
          return ::std::any(__fn());
        }
      };

      if constexpr (::cuda::std::is_void_v<_Raw>)
      {
        __check_compatible<_Raw>();
        static_cast<void>(__p_->hook(__e, __loc, __proxy));
        return ::std::ignore;
      }
      else if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Raw>, ::std::any>)
      {
        return __p_->hook(__e, __loc, __proxy);
      }
      else
      {
        __check_compatible<_Raw>();
        return __unbox<_Raw>(__p_->hook(__e, __loc, __proxy));
      }
    }
  }

  //! @brief Entry gate: forwards to the erased policy (no-op when it has none).
  void on_enter()
  {
    __p_->on_enter();
  }

  //! @brief Type-preserving success passthrough: box, delegate, unbox to the same type.
  template <class _R>
  _R on_success(_R&& __r)
  {
    static_assert(!::cuda::std::is_reference_v<_R>,
                  "exception_sink cannot serve a reference-returning callable: std::any cannot "
                  "carry references; use a concrete policy, or return a pointer");
    if constexpr (::cuda::std::is_reference_v<_R>)
    {
      _CCCL_UNREACHABLE();
    }
    else
    {
      __check_compatible<_R>();
      return __unbox<_R>(__p_->on_success(::std::any(::cuda::std::forward<_R>(__r))));
    }
  }
  void on_success()
  {
    static_cast<void>(__p_->on_success());
  }
};

//! @brief Erase any policy (or historical reaction) into an @ref exception_sink.
template <class _P>
exception_sink type_erase(_P&& __p)
{
  return exception_sink{detail::__normalize(::cuda::std::forward<_P>(__p))};
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
 * @param[in] __reaction The policy (or a reaction normalized into one), owned if passed an
 *            rvalue and referred to if passed an lvalue.
 * @param[in] __loc The location passed to exception hooks.
 * @return A policy object consumed by `operator<<`.
 */
template <class _Reaction>
auto on_throw(_Reaction&& __reaction,
              const ::cuda::std::source_location __loc = ::cuda::std::source_location::current()) noexcept
{
  return exception_policies::detail::__on_throw_policy{
    exception_policies::detail::__normalize(::cuda::std::forward<_Reaction>(__reaction)), __loc};
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
//!
//! The block-scope using-declarations pin `abort` and `terminate` to the
//! policies. Without them, using the macro at global scope with the C
//! library in scope would be ambiguous: the using-directive parks the
//! policy names at the nearest namespace enclosing both it and
//! `exception_policies`, which in user code is the global namespace,
//! right next to the C library's `abort`.
#define ON_THROW(...)                                               \
  [&] {                                                             \
    using namespace ::cuda::experimental::stf::exception_policies;  \
    using ::cuda::experimental::stf::exception_policies::abort;     \
    using ::cuda::experimental::stf::exception_policies::terminate; \
    return ::cuda::experimental::stf::on_throw(__VA_ARGS__);        \
  }()                                                               \
    << [&]()

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
  bool __gated = false;
  _CCCL_TRY
  {
    on_throw(guarded) << flaky;
  }
  _CCCL_CATCH ([[maybe_unused]] const pol::circuit_open& __open)
  {
    __gated = true;
  }
  _CCCL_CATCH_ALL
  {
    EXPECT(false, "the gate must refuse with circuit_open, nothing else");
  }
  EXPECT(__gated);
  EXPECT(runs == 2);

  // External administration: refill through the shared int, then a success restores the
  // budget to its creation-time value.
  *budget = 1;
  EXPECT((on_throw(guarded) << []() -> int {
           return 7;
         })
         == 7);
  EXPECT(*budget == 2);

  // The macro spelling gates identically.
  *budget = 0;
  __gated = false;
  _CCCL_TRY
  {
    ON_THROW(circuit_breaker(budget) & subst(-1))
    {
      return 9;
    };
  }
  _CCCL_CATCH ([[maybe_unused]] const pol::circuit_open& __open)
  {
    __gated = true;
  }
  _CCCL_CATCH_ALL
  {
    EXPECT(false, "the gate must refuse with circuit_open, nothing else");
  }
  EXPECT(__gated);

  // The erased form carries the gate through: sinks re-erase, gates survive.
  *budget                      = 0;
  pol::exception_sink __erased = pol::type_erase(pol::circuit_breaker(budget) & pol::subst(-1));
  __gated                      = false;
  _CCCL_TRY
  {
    on_throw(__erased) << flaky;
  }
  _CCCL_CATCH ([[maybe_unused]] const pol::circuit_open& __open)
  {
    __gated = true;
  }
  _CCCL_CATCH_ALL
  {
    EXPECT(false, "the erased gate must refuse with circuit_open, nothing else");
  }
  EXPECT(__gated);
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

UNITTEST("defer across threads")
{
  using namespace cuda::experimental::stf;
  namespace pol = cuda::experimental::stf::exception_policies;

  // One worker fails; the exception travels as a value. The worker is joined (and its
  // thread object gone) before handling: the exception_ptr alone keeps the exception
  // alive, so the handler may outlive the worker by any margin.
  ::std::exception_ptr __slot;
  ::std::thread __worker([&__slot] {
    __slot = on_throw(pol::defer) << []() -> ::std::exception_ptr {
      throw ::std::runtime_error("worker failed");
    };
  });
  __worker.join();
  EXPECT(__slot != nullptr);

  bool __handled = false;
  ::std::thread __handler([&__slot, &__handled] {
    _CCCL_TRY
    {
      ::std::rethrow_exception(__slot);
    }
    _CCCL_CATCH ([[maybe_unused]] const ::std::runtime_error& __e)
    {
      __handled = true;
    }
    _CCCL_CATCH_ALL {}
  });
  __handler.join();
  EXPECT(__handled);

  // N workers, one handler: even workers fail, odd workers answer "no exception".
  constexpr int __n = 4;
  ::std::exception_ptr __slots[__n];
  ::std::thread __workers[__n];
  for (int __i = 0; __i < __n; ++__i)
  {
    __workers[__i] = ::std::thread([&__slots, __i] {
      __slots[__i] = on_throw(pol::defer) << [__i]() -> ::std::exception_ptr {
        if (__i % 2 == 0)
        {
          throw ::std::logic_error("even worker");
        }
        return ::std::exception_ptr{};
      };
    });
  }
  for (auto& __t : __workers)
  {
    __t.join();
  }
  int __failures = 0;
  for (auto& __s : __slots)
  {
    if (__s)
    {
      _CCCL_TRY
      {
        ::std::rethrow_exception(__s);
      }
      _CCCL_CATCH ([[maybe_unused]] const ::std::logic_error& __e)
      {
        ++__failures;
      }
      _CCCL_CATCH_ALL {}
    }
  }
  EXPECT(__failures == 2);
};

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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("policy algebra")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
#  if _CCCL_HAS_EXCEPTIONS()
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
  // potentially throwing (the __builtin_LINE machinery), which would taint the query with
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("policy inventory")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
#  if _CCCL_HAS_EXCEPTIONS()
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
  using _Result   = ::cuda::std::expected<int, ::std::exception_ptr>;
  const auto good = on_throw(as_expected) << []() -> _Result {
    return 5; // expected's converting constructor keeps the happy path natural
  };
  static_assert(::cuda::std::is_same_v<decltype(good), const _Result>,
                "the callable owns the as_expected expression type");
  EXPECT(good.has_value());
  // Dereference rather than .value(): value() would instantiate bad_expected_access, whose
  // inlined exception_ptr destructor trips a spurious gcc 14/15 -O3 maybe-uninitialized in
  // every TU that compiles these tests.
  EXPECT(*good == 5);

  const auto bad = on_throw(as_expected) << []() -> _Result {
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("re-running policies")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if _CCCL_HAS_EXCEPTIONS()
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
    using _Result = ::cuda::std::expected<int, ::std::exception_ptr>;
    int calls     = 0;
    const auto r  = on_throw(as_expected & retry) << [&]() -> _Result {
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
    const int v = on_throw([](const ::std::exception*, auto, auto& __fn) {
                    return __fn();
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

// Helper exception types for the tests below, at namespace scope on purpose: nvcc <= 12.9 in
// C++20 mode infers __host__ __device__ for a local class's special members inside an extended
// lambda, and the inherited std::runtime_error constructor is host-only (error #20011).
struct __ut_error_from_exception
{
  int marker;

  explicit __ut_error_from_exception(const ::std::exception& __exception)
      : marker(__exception.what()[0])
  {}
};
struct __ut_low_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};
struct __ut_high_error : ::std::runtime_error
{
  using ::std::runtime_error::runtime_error;
};

UNITTEST("as_expected and defer")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if _CCCL_HAS_EXCEPTIONS()
  using _PtrResult = ::cuda::std::expected<int, ::std::exception_ptr>;
  using _RefResult = ::cuda::std::expected<int, __ut_error_from_exception>;
  static_assert(detail::__is_expected<_PtrResult>);
  static_assert(detail::__is_expected<_RefResult>);
  static_assert(!detail::__is_expected<int>);

  // The callable declares the boundary type; expected's converting constructor keeps a bare
  // success return natural.
  {
    const auto r = on_throw(as_expected) << []() -> _PtrResult {
      return 42;
    };
    static_assert(::cuda::std::is_same_v<decltype(r), const _PtrResult>);
    EXPECT(r.has_value());
    EXPECT(*r == 42);
  }

  // First ladder rung: the error type accepts the active exception_ptr.
  {
    const auto r = on_throw(as_expected) << []() -> _PtrResult {
      throw ::std::runtime_error("captured");
    };
    EXPECT(!r.has_value());
    bool rethrown = false;
    try
    {
      ::std::rethrow_exception(r.error());
    }
    catch (const ::std::runtime_error& __exception)
    {
      rethrown = ::std::string_view{__exception.what()} == "captured";
    }
    EXPECT(rethrown);
  }

  // Second ladder rung: construct the error from the funneled std::exception.
  {
    const auto r = on_throw(as_expected) << []() -> _RefResult {
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
      on_throw(as_expected) << []() -> _RefResult {
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
    catch (const ::std::logic_error& __exception)
    {
      rethrown = ::std::string_view{__exception.what()} == "deferred";
    }
    EXPECT(rethrown);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(as_expected) << []() -> int { return 1; };
  //      -> "as_expected requires the callable to return a cuda::std::expected instantiation"
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("guard translate delay backoff remember")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;

#  if _CCCL_HAS_EXCEPTIONS()
  // A true guard contributes an effect-only answer; a false guard declines its whole sequence.
  {
    const auto is_low = [](const ::std::exception* __exception) {
      return __exception && dynamic_cast<const __ut_low_error*>(__exception);
    };
    const int claimed = on_throw(when(is_low, subst(1)) | subst(2)) << []() -> int {
      throw __ut_low_error("low");
    };
    const int declined = on_throw(when(is_low, subst(1)) | subst(2)) << []() -> int {
      throw __ut_high_error("high");
    };
    EXPECT(claimed == 1);
    EXPECT(declined == 2);
  }
  {
    const int claimed =
      on_throw(when(
                 [](const ::std::exception* __exception) {
                   return !__exception;
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
      throw __ut_low_error("low");
    };
    const int declined =
      on_throw(when(
                 [] {
                   return false;
                 },
                 subst(3))
               | subst(4))
      << []() -> int {
      throw __ut_low_error("low");
    };
    EXPECT(accepted == 3);
    EXPECT(declined == 4);
  }

  // translate<From, To>: a From becomes a To for the next typed arm; non-From declines.
  {
    const int v = on_throw(translate<__ut_low_error, __ut_high_error> | catch_only<__ut_high_error>(subst(1)))
               << []() -> int {
      throw __ut_low_error("cause");
    };
    EXPECT(v == 1);
    const int passed = on_throw(translate<__ut_low_error, __ut_high_error> | subst(2)) << []() -> int {
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
      on_throw(nest(__ut_high_error{"context"})) << [] {
        throw __ut_low_error("cause");
      };
    }
    catch (const __ut_high_error& __exception)
    {
      saw_high = true;
      try
      {
        ::std::rethrow_if_nested(__exception);
      }
      catch (const __ut_low_error&)
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
      throw __ut_low_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }

  // Backoff owns its retry loop: exhaustion declines, while an early success answers.
  {
    int calls   = 0;
    const int v = on_throw(backoff(2, ::std::chrono::milliseconds{1}) | subst(-1)) << [&]() -> int {
      ++calls;
      throw __ut_low_error("always");
    };
    EXPECT(v == -1);
    EXPECT(calls == 3);
  }
  {
    int calls   = 0;
    const int v = on_throw(backoff(2, ::std::chrono::milliseconds{1})) << [&]() -> int {
      if (++calls == 1)
      {
        throw __ut_low_error("once");
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
        throw __ut_low_error("once");
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
      throw __ut_low_error("offline");
    };
    EXPECT(stale == 7);

    const int fresh = ON_THROW(remember(&last))
    {
      return 9;
    };
    const int served = ON_THROW(remember(&last))->int
    {
      throw __ut_low_error("offline");
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
      throw __ut_low_error("offline");
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("repetition")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
  static_assert(::cuda::std::is_empty_v<retry_t>, "retry is the stateless unit re-attempt");

#  if _CCCL_HAS_EXCEPTIONS()
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
    using _Result = ::cuda::std::expected<int, ::std::exception_ptr>;
    int calls     = 0;
    const auto r  = on_throw((as_expected & retry) * 3) << [&]() -> _Result {
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("ON_THROW macro")
{
  using namespace cuda::experimental::stf; // deliberately NOT exception_policies:
                                           // the macro must supply the vocabulary
#  if _CCCL_HAS_EXCEPTIONS()
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("type erasure")
{
  using namespace cuda::experimental::stf;
  using namespace cuda::experimental::stf::exception_policies;
#  if _CCCL_HAS_EXCEPTIONS()
  const auto __names_both = [](const ::std::logic_error& __err, ::std::string_view __a, ::std::string_view __b) {
    const ::std::string __msg{__err.what()};
    return __msg.find(::std::string{__a}) != ::std::string::npos
        && __msg.find(::std::string{__b}) != ::std::string::npos;
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
    catch (const ::std::logic_error& __err)
    {
      failed = __names_both(__err, type_name<long long>, type_name<int>);
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
    catch (const ::std::logic_error& __err)
    {
      failed = __names_both(__err, type_name<double>, type_name<int>);
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
    catch (const ::std::logic_error& __err)
    {
      failed = __names_both(__err, type_name<::std::exception_ptr>, type_name<int>);
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
    catch (const ::std::logic_error& __err)
    {
      failed = ::std::string_view{__err.what()}.find(type_name<unsigned>) != ::std::string_view::npos;
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
    catch (const ::std::logic_error& __err)
    {
      failed = ::std::string_view{__err.what()}.find(type_name<int>) != ::std::string_view::npos;
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

UNITTEST("policies inside handlers")
{
  using namespace cuda::experimental::stf;
  namespace pol = cuda::experimental::stf::exception_policies;

  // SCOPE(fail) declared inside a catch block fires only when the HANDLER throws, never
  // for the exception being handled. The relative uncaught_exceptions() idiom guarantees
  // it; this is the classic case that broke bool uncaught_exception().
  bool __fail_fired = false;
  _CCCL_TRY
  {
    throw 1;
  }
  _CCCL_CATCH_ALL
  {
    SCOPE(fail)
    {
      __fail_fired = true;
    };
  }
  EXPECT(!__fail_fired);

  _CCCL_TRY
  {
    _CCCL_TRY
    {
      throw 1;
    }
    _CCCL_CATCH_ALL
    {
      SCOPE(fail)
      {
        __fail_fired = true;
      };
      throw 2;
    }
  }
  _CCCL_CATCH_ALL {}
  EXPECT(__fail_fired);

  // SCOPE(success) inside a handler: the dual, fires on the handler's normal exit.
  bool __success_fired = false;
  _CCCL_TRY
  {
    throw 1;
  }
  _CCCL_CATCH_ALL
  {
    SCOPE(success)
    {
      __success_fired = true;
    };
  }
  EXPECT(__success_fired);

  // on_throw composes inside a handler; rethrow's bare `throw;` inside the inner
  // expression's catch is legal while an outer exception is also being handled.
  int __v = 0;
  _CCCL_TRY
  {
    throw ::std::runtime_error("outer");
  }
  _CCCL_CATCH_ALL
  {
    __v = on_throw(pol::rethrow | pol::subst(7)) << []() -> int {
      throw ::std::logic_error("inner");
    };
  }
  EXPECT(__v == 7);

  // on_throw(defer) inside a handler captures the exception ITS BODY threw (current
  // inside its own catch), never the exception the surrounding handler is handling.
  _CCCL_TRY
  {
    throw ::std::runtime_error("outer");
  }
  _CCCL_CATCH_ALL
  {
    auto __ep = on_throw(pol::defer) << []() -> ::std::exception_ptr {
      throw ::std::logic_error("inner");
    };
    bool __got_inner = false;
    _CCCL_TRY
    {
      ::std::rethrow_exception(__ep);
    }
    _CCCL_CATCH ([[maybe_unused]] const ::std::logic_error& __e)
    {
      __got_inner = true;
    }
    _CCCL_CATCH_ALL {}
    EXPECT(__got_inner);
  }
};

#endif // UNITTESTED_FILE
