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
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
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
#include <typeinfo>
#include <utility>

#ifdef UNITTESTED_FILE
#  include <sstream>
#  include <string>
#endif // UNITTESTED_FILE

// nvcc 12.0 hits an internal compiler error ("error while padding end of structure!") when a
// [[no_unique_address]] member's type is one of this header's empty policies; newer toolkits
// are fine. WAR: the attribute is applied only where the compiler survives it.
#if _CCCL_CUDA_COMPILER(NVCC, <, 12, 1)
#  define _CCCL_STF_NO_UNIQUE_ADDRESS
#else // ^^^ nvcc < 12.1 ^^^ / vvv other compilers vvv
#  define _CCCL_STF_NO_UNIQUE_ADDRESS _CCCL_NO_UNIQUE_ADDRESS
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
 * `on_throw(abort) << callable`, in ternaries (`ready ? front() : abort()`), and as a bare
 * call `abort()`.
 *
 * Inside this namespace, plain `abort` finds this object before the C library's function. Code
 * that sees both through using-directives gets an ambiguity error rather than a silent pick,
 * and disambiguates with a using-declaration: `using cuda::experimental::stf::abort;`. A
 * block-scope using-declaration still hides `::abort`.
 *
 * `notify & abort` reports twice (documented). `abort | p` is a dead-| error (hook is
 * noexcept); `abort & p` is a dead-& error (answers `nothing`).
 */
struct abort_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  //! @brief The bare call: usable in ternaries -- `ready ? front() : abort()`.
  [[noreturn]] nothing operator()() const noexcept
  {
    ::std::abort();
  }

  //! @brief The exception hook: report, then die.
  template <class _Fn>
  [[noreturn]] nothing
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

  [[noreturn]] nothing operator()() const noexcept
  {
    ::std::terminate();
  }

  template <class _Fn>
  [[noreturn]] nothing
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
  using __exception_sink_tag = void;
  //! @endcond

  //! @brief Captures the in-flight exception; the answer converts to the expression's type.
  template <class _Fn>
  ::std::exception_ptr operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const noexcept
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
  using __exception_sink_tag = void;
  //! @endcond

  template <class _Fn>
  [[noreturn]] nothing operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const
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

  _CCCL_STF_NO_UNIQUE_ADDRESS _V __v_;

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
  // not to a copy (the ignore branch deduces a reference to the global, which never dangles).
  template <class _Fn>
  decltype(auto) operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn& __fn)
  {
    if constexpr (::cuda::std::is_void_v<decltype(__fn())>)
    {
      __fn(); // a throw here IS the decline
      return (::std::ignore); // resume: the void expression is complete
    }
    else
    {
      return __fn();
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
template <class _E>
struct expecting_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  template <class _R>
  ::cuda::std::expected<::cuda::std::decay_t<_R>, _E> on_success(_R&& __r) const
  {
    return ::cuda::std::expected<::cuda::std::decay_t<_R>, _E>{::cuda::std::in_place, ::cuda::std::forward<_R>(__r)};
  }

  template <class _Void = void>
  ::cuda::std::expected<_Void, _E> on_success() const
  {
    return ::cuda::std::expected<_Void, _E>{};
  }

  template <class _Fn>
  ::cuda::std::unexpected<_E> operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const
  {
    _CCCL_TRY
    {
      throw; // re-observe the active exception at type _E
    }
    _CCCL_CATCH (const _E& __caught)
    {
      if constexpr (::cuda::std::is_polymorphic_v<_E>)
      {
        if (typeid(__caught) != typeid(_E))
        {
          throw; // a derivative: decline rather than slice
        }
      }
      return ::cuda::std::unexpected<_E>{__caught}; // by value, no allocation
    }
    _CCCL_CATCH_FALLTHROUGH // no catch-all: an unmatched rethrow propagates, which IS the decline
    _CCCL_UNREACHABLE();
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
  using __exception_sink_tag = void;
  //! @endcond

  template <class _R>
  ::cuda::std::expected<::cuda::std::decay_t<_R>, ::std::exception_ptr> on_success(_R&& __r) const
  {
    return ::cuda::std::expected<::cuda::std::decay_t<_R>, ::std::exception_ptr>{
      ::cuda::std::in_place, ::cuda::std::forward<_R>(__r)};
  }

  // Templates in name only: laziness keeps expected<void, exception_ptr> (and, through it,
  // bad_expected_access) from being instantiated in every including TU -- gcc 14/15 at -O3
  // report a spurious maybe-uninitialized inside the latter's inlined destructor.
  template <class _Void = void>
  ::cuda::std::expected<_Void, ::std::exception_ptr> on_success() const
  {
    return ::cuda::std::expected<_Void, ::std::exception_ptr>{};
  }

  template <class _Fn, class _Eptr = ::std::exception_ptr>
  ::cuda::std::unexpected<_Eptr>
  operator()(const ::std::exception*, const ::cuda::std::source_location, _Fn&) const noexcept
  {
    return ::cuda::std::unexpected<_Eptr>{::std::current_exception()};
  }
};

template <class _E>
inline constexpr expecting_t<_E> expecting{};

//! @brief Baseline instance of `expecting<std::exception_ptr>`; see @ref expecting_t.
inline constexpr expecting_t<::std::exception_ptr> as_expected{};

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

// Whether a policy's answer type is `nothing` -- it never returns from the exception path. The
// two-step form keeps `__hook_answer_t` from being named for a hookless policy: `&&` does not
// short-circuit template instantiation, so the answer is probed only in the `true` partial.
template <bool _HasHook, class _P, class _Fn>
inline constexpr bool __answers_nothing_impl = false;

template <class _P, class _Fn>
inline constexpr bool __answers_nothing_impl<true, _P, _Fn> =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<__hook_answer_t<_P, _Fn>>, nothing>;

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
// `__as_policy`, `__policy_pow`): both arities delegate to the wrapped policy, which owns
// the expression type.
template <class _P>
struct __forwards_success
{
  _CCCL_STF_NO_UNIQUE_ADDRESS _P __p_;

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
};

// Intra-pack subsumption: reject when any listed type is derived-or-equal to another
// (`is_base_of_v<A, A>` catches duplicates). Message names the dead Derived entry.
template <class...>
inline constexpr bool __catch_only_pack_ok = true;

template <class _Head, class... _Tail>
inline constexpr bool __catch_only_pack_ok<_Head, _Tail...> =
  (!::cuda::std::is_base_of_v<_Head, _Tail> && ...) && (!::cuda::std::is_base_of_v<_Tail, _Head> && ...)
  && __catch_only_pack_ok<_Tail...>;

// `catch_only<E1, E2, ...>(p)`: run `p`'s exception path when the exception matches ANY listed
// type (derived-or-equal, dynamic_cast semantics), else decline by rethrowing. Native C++ has
// no multi-type catch clause; this adds expressivity the language lacks. Policy parameter
// leads so the exception-type pack trails.
template <class _P, class... _Es>
struct __catch_only_t : __forwards_success<_P>
{
  using __exception_sink_tag = void;

  template <class _Fn, class _Self = _P, ::cuda::std::enable_if_t<__has_exception_hook<_Self>, int> = 0>
  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
  {
    if (__exception && (... || dynamic_cast<const _Es*>(__exception)))
    {
      return this->__p_(__exception, __loc, __fn);
    }
    throw; // decline: wrong type, or a non-std exception (nullptr)
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

// Success-hook forwarding shared by the `&` and `|` composites: rightmost-wins (the right
// element owns the expression type). The exception hook -- where the two composites differ --
// lives in the derived types.
template <class _L, class _R>
struct __composite_hooks
{
  _CCCL_STF_NO_UNIQUE_ADDRESS _L __l_;
  _CCCL_STF_NO_UNIQUE_ADDRESS _R __r_;

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
      static_cast<void>(this->__l_(__exception, __loc, __fn));
      if constexpr (__has_exception_hook<_R>)
      {
        return this->__r_(__exception, __loc, __fn);
      }
    }
    else
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

// The alternation composite `_L | _R`: `_L` claims first; if it declines by throwing, `_R`
// handles the original (re-observed) exception. Each arm is called at the uniform 3-arg shape;
// acceptance is interpreted at `decltype(fn())`. A plain arm's answer must therefore be
// interpretable at the callable's result type -- `retry | subst(-1)` works, but
// `retry | as_expected` does not (an `unexpected` does not convert to the raw result);
// nest `on_throw(as_expected) << [&]{ return on_throw(retry * n) << f; }` for that spelling.
template <class _L, class _R>
struct __policy_or : __composite_hooks<_L, _R>
{
  using __exception_sink_tag = void;

  static_assert(__has_exception_hook<_L> && __has_exception_hook<_R>,
                "both sides of | must answer the exception path (have an exception hook)");
  static_assert(!__exception_path_nothrow_v<_L>,
                "the left policy never declines; alternatives after it are unreachable");

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

    // When the left arm owns the expression type (on_success), both arms answer in that owned
    // type: a match wraps the left's unexpected, a decline lifts the right arm's raw value
    // through on_success. This is what makes `expecting<E> | subst(v)` work.
    if constexpr (__has_on_success_with<_L, _Raw>)
    {
      using _Owned = decltype(::cuda::std::declval<_L&>().on_success(::cuda::std::declval<_Raw>()));
      _CCCL_TRY
      {
        return _Owned{this->__l_(__exception, __loc, __fn)};
      }
      _CCCL_CATCH_ALL
      {
        return this->__l_.on_success(__reobserve_right());
      }
    }
    else if constexpr (::cuda::std::is_void_v<_Raw> && __has_on_success_void<_L>)
    {
      using _Owned = decltype(::cuda::std::declval<_L&>().on_success());
      _CCCL_TRY
      {
        return _Owned{this->__l_(__exception, __loc, __fn)};
      }
      _CCCL_CATCH_ALL
      {
        __reobserve_right();
        return this->__l_.on_success();
      }
    }
    else
    {
      // Neither arm owns: interpret each at the callable's result type. Void callables surface
      // ignore so this composite can still sit as a top-level policy.
      _CCCL_TRY
      {
        if constexpr (::cuda::std::is_void_v<_Raw>)
        {
          __interpret_answer<_Raw>(this->__l_, __exception, __loc, __fn);
          return (::std::ignore);
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
          return (::std::ignore);
        }
        else
        {
          return __reobserve_right();
        }
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
      return (::std::ignore);
    }
    else
    {
      return __go(__go, __exception, __n_);
    }
    _CCCL_DIAG_POP
  }
};

// Interpret the final element's answer as the expression's value, converting to `_Expr`.
template <class _Expr, class _P, class _Fn>
_Expr __interpret_answer(
  _P& __policy, const ::std::exception* __exception, const ::cuda::std::source_location __loc, _Fn& __fn)
{
  using _Answer = __hook_answer_t<_P, _Fn>;
  static_assert(!::cuda::std::is_void_v<_Answer>,
                "the final policy must answer the exception path: nothing to die, ::std::ignore "
                "to resume, or a value to substitute");

  if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Answer>, nothing>)
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
                  "nothing, like abort and terminate), ::std::ignore, or a value convertible to "
                  "the result of the callable");
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
  _CCCL_STF_NO_UNIQUE_ADDRESS _Reaction __reaction_;
  const ::cuda::std::source_location __loc_;
};

template <class _R>
__on_throw_policy(_R, ::cuda::std::source_location) -> __on_throw_policy<_R>;

template <class _Reaction, class _Fn>
// A resuming chain reads neither exception nor location in some instantiations; gcc 9 flags the
// unread policy without the attribute.
decltype(auto) operator<<([[maybe_unused]] __on_throw_policy<_Reaction> __policy,
                          _Fn&& __fn) noexcept(__exception_path_nothrow_v<_Reaction, _Fn>)
{
  // Bind as a non-const lvalue: a hook may invoke it again later.
  _Fn& __f = __fn;

  // A `noexcept` callable puts the policy out of reach: an exception raised inside it ends the
  // program where it stands, so the catch below could never run and the policy would be a
  // promise nobody keeps.
  static_assert(!noexcept(__f()),
                "on_throw has nothing to do for a noexcept callable, which terminates rather than "
                "throws; call such a callable directly");

  using _Result = decltype(__f());
  using _P      = _Reaction;

  if constexpr (::cuda::std::is_void_v<_Result>)
  {
    if constexpr (__has_on_success_void<_P>)
    {
      // A success hook owns the type (e.g. defer, as_expected): the expression is what it makes.
      using _Expr = decltype(::cuda::std::declval<_P&>().on_success());
      _CCCL_TRY
      {
        __f();
        return __policy.__reaction_.on_success();
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
      // No success hook: the expression is void, the result passing through.
      _CCCL_TRY
      {
        __f();
      }
      _CCCL_CATCH (const ::std::exception& __exception)
      {
        return detail::__on_exception<void>(__policy.__reaction_, &__exception, __policy.__loc_, __f);
      }
      _CCCL_CATCH_ALL
      {
        return detail::__on_exception<void>(__policy.__reaction_, nullptr, __policy.__loc_, __f);
      }
    }
  }
  else if constexpr (__has_on_success_with<_P, _Result>)
  {
    // A success hook transforms/owns the non-void result; the expression is its return type.
    using _Expr = decltype(::cuda::std::declval<_P&>().on_success(::cuda::std::declval<_Result>()));
    _CCCL_TRY
    {
      return __policy.__reaction_.on_success(__f());
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
    // No hook accepts the non-void result: pass it through, unless a void-only success hook
    // (e.g. defer over a non-void callable) means the result has no channel.
    static_assert(!__has_on_success_void<_P>, "the policy's on_success cannot accept the callable's result");
    _CCCL_TRY
    {
      return __f();
    }
    _CCCL_CATCH (const ::std::exception& __exception)
    {
      return detail::__on_exception<_Result>(__policy.__reaction_, &__exception, __policy.__loc_, __f);
    }
    _CCCL_CATCH_ALL
    {
      return detail::__on_exception<_Result>(__policy.__reaction_, nullptr, __policy.__loc_, __f);
    }
  }
}
} // namespace detail
#endif // !_CCCL_DOXYGEN_INVOKED

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
 * a success hook `on_success(...)` that observes or replaces the result. The named policies are
 * @ref notify_t "notify", @ref subst_t "subst", @ref defer_t "defer", @ref rethrow_t "rethrow",
 * @ref retry_t "retry", @ref expecting_t "expecting" / @ref as_expected, @ref noop_t "noop", and
 * @ref catch_only. Policies compose with `&` (sequence; the last element answers; non-final
 * answers are discarded) and `|` (alternation; the left may decline by throwing), and with
 * `*` (n-fold `|`).
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
  return detail::__on_throw_policy{detail::__normalize(::cuda::std::forward<_Reaction>(__reaction)), __loc};
}

/**
 * @brief Restricts a policy to exceptions matching any of `E1, E2, ...`: `catch_only<E...>(p)`
 * runs `p`'s exception path when the caught exception is derived-or-equal to any listed type
 * (dynamic_cast semantics), and otherwise declines by rethrowing. Native C++ has no multi-type
 * catch clause; this adds that expressivity. A non-`std::exception` (nullptr) always declines.
 * Each `E` must derive from `std::exception`. A pack where one type subsumes another is
 * rejected (the derived entry would be dead).
 */
template <class... _Es, class _P>
auto catch_only(_P&& __p)
{
  static_assert(sizeof...(_Es) > 0, "catch_only requires at least one exception type");
  static_assert((::cuda::std::is_base_of_v<::std::exception, _Es> && ...),
                "catch_only<E...> requires every E to derive from std::exception");
  static_assert(detail::__catch_only_pack_ok<_Es...>,
                "catch_only<..., Base, ..., Derived, ...>: the Derived entry is dead "
                "(Base already claims it)");
  auto __np = detail::__normalize(::cuda::std::forward<_P>(__p));
  return detail::__catch_only_t<decltype(__np), _Es...>{::cuda::std::move(__np)};
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

// Negative-compile expectations (do not compile; kept as comments near the code they guard):
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
  //! [on_throw]
  // The C library also declares ::abort, so under a using-directive the typed one is picked
  // by name; qualifying every use works as well.
  using cuda::experimental::stf::abort;
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("policy algebra")
{
  using namespace cuda::experimental::stf;
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
  static_assert(::cuda::std::is_same_v<decltype(noop & abort), abort_t>,
                "abort is a policy; elimination returns it bare");
  {
    using cuda::experimental::stf::abort; // block-scope: hides ::abort
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
  //  - on_throw(subst(8) | subst(9)) << []() -> int { throw 1; };
  //      -> "the left policy never declines; alternatives after it are unreachable"
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("policy inventory")
{
  using namespace cuda::experimental::stf;
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
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("re-running policies")
{
  using namespace cuda::experimental::stf;

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
  //  - on_throw(retry | as_expected) << []() -> int { ... };
  //      -> conversion failure: an owner's unexpected answer does not convert to the
  //         callable's result; owners belong at the top (or left of &), not in | arms
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("expecting")
{
  using namespace cuda::experimental::stf;

#  if _CCCL_HAS_EXCEPTIONS()
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
    struct derived_error : ::std::runtime_error
    {
      using ::std::runtime_error::runtime_error;
    };
    bool escaped = false;
    try
    {
      on_throw(expecting<::std::runtime_error>) << [&]() -> int {
        throw derived_error{"sliced?"};
      };
    }
    catch (const derived_error&)
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
    struct my_error
    {
      int code;
    };
    const auto r = on_throw(expecting<my_error>) << []() -> int {
      throw my_error{7};
    };
    EXPECT(!r.has_value());
    EXPECT(r.error().code == 7);
  }
  // A nonstandard polymorphic hierarchy still gets exact matching (derivative declines).
  {
    struct poly_base
    {
      virtual ~poly_base() = default;
    };
    struct poly_derived : poly_base
    {};
    bool escaped = false;
    try
    {
      on_throw(expecting<poly_base>) << []() -> int {
        throw poly_derived{};
      };
    }
    catch (const poly_derived&)
    {
      escaped = true;
    }
    EXPECT(escaped);
  }

  // Negative-compile expectations (do not compile; kept as comments near the code they guard):
  //  - on_throw(expecting<::std::exception_ptr> | subst(0)) << ...;
  //      -> "the left policy never declines; ..." (the total form can't head a ladder)
#  endif // _CCCL_HAS_EXCEPTIONS()
};

UNITTEST("repetition")
{
  using namespace cuda::experimental::stf;
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
