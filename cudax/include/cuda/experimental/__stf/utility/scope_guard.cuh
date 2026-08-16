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
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
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
#include <utility>

#ifdef UNITTESTED_FILE
#  include <sstream>
#  include <string>
#endif // UNITTESTED_FILE

namespace cuda::experimental::stf
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

  // Destination: at most one of these is set. Both null means `stderr`.
  ::FILE* __file_       = nullptr;
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
  //! else the configured `FILE*` if any, else `stderr`. The ostream write is best-effort: a
  //! stream configured to throw does not get to end the program from inside a handler.
  decltype(::std::ignore)
  operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc) const noexcept
  {
    if (__os_ != nullptr)
    {
      _CCCL_TRY
      {
        *__os_ << __loc.file_name() << '(' << __loc.line() << ") on_throw violation in " << __loc.function_name()
               << ": " << (__exception != nullptr ? __exception->what() : "nonstandard exception") << '\n';
        __os_->flush();
      }
      _CCCL_CATCH_ALL {}
    }
    else
    {
      ::FILE* const __out = __file_ != nullptr ? __file_ : stderr;
      ::fprintf(__out,
                "%s(%u) on_throw violation in %s: %s\n",
                __loc.file_name(),
                __loc.line(),
                __loc.function_name(),
                __exception != nullptr ? __exception->what() : "nonstandard exception");
      ::fflush(__out);
    }
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
  ::std::exception_ptr operator()([[maybe_unused]] const ::std::exception* __exception,
                                  [[maybe_unused]] const ::cuda::std::source_location __loc) const noexcept
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

  [[noreturn]] nothing operator()([[maybe_unused]] const ::std::exception* __exception,
                                  [[maybe_unused]] const ::cuda::std::source_location __loc) const
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
 * `(const std::exception*, source_location)` when that is well-formed (so
 * `subst([](const std::exception* e, auto){ ... })` reacts to the exception); else the result
 * of invoking it as a nullary callable, a lazy fallback computed only on the exception path
 * (`subst([]{ return expensive(); })`); else the stored value itself, forwarded out and owned
 * if the policy owns it, referred to if it holds an lvalue reference (so a replacement passed
 * as an lvalue can stand in for a reference result).
 */
template <class _V>
struct subst_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  _V __v_;

  decltype(auto) operator()([[maybe_unused]] const ::std::exception* __exception,
                            [[maybe_unused]] const ::cuda::std::source_location __loc) noexcept
  {
    if constexpr (::cuda::std::is_invocable_v<_V&, const ::std::exception*, ::cuda::std::source_location>)
    {
      return __v_(__exception, __loc);
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
 * @brief Around-hook policy: re-invokes the callable up to `n` times after the first failure
 * (`n + 1` attempts at most), propagating the last exception if all fail.
 *
 * `retry` has no exception hook of its own, so `on_throw(retry(3)) << f` retries and then
 * propagates; pair it with a terminal, as in `retry(3) & subst(fallback)`, to answer the
 * exhausted failure. The counter is mutable state; policies are stored by value per `on_throw`
 * expression, so each `<<` gets its own count.
 */
struct retry_t
{
  //! @cond
  using __exception_sink_tag = void;
  //! @endcond

  int __left_;

  template <class _Fn>
  decltype(auto) invoke(_Fn&& __fn)
  {
    for (;; --__left_)
    {
      _CCCL_TRY
      {
        return ::cuda::std::forward<_Fn>(__fn)();
      }
      _CCCL_CATCH_ALL
      {
        if (__left_ == 0)
        {
          throw;
        }
      }
    }
  }
};

//! @brief Creates a retrying policy; see @ref retry_t.
inline retry_t retry(int __n)
{
  // A negative count would decrement past the `== 0` stop and retry effectively forever.
  _CCCL_ASSERT(__n >= 0, "retry requires a non-negative count");
  return retry_t{__n};
}

/**
 * @brief Success-and-exception policy that turns the outcome into a
 * `cuda::std::expected<R, std::exception_ptr>`: the callable's result on success, the captured
 * exception on failure. A `void` callable yields `cuda::std::expected<void, exception_ptr>`.
 *
 * The exception hook cannot know `R`, so it answers with a `cuda::std::unexpected` and relies
 * on the conversion into the expression type (an `expected` is constructible from a matching
 * `unexpected`).
 */
struct as_expected_t
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

  template <class _Eptr = ::std::exception_ptr>
  ::cuda::std::unexpected<_Eptr> operator()([[maybe_unused]] const ::std::exception* __exception,
                                            [[maybe_unused]] const ::cuda::std::source_location __loc) const noexcept
  {
    return ::cuda::std::unexpected<_Eptr>{::std::current_exception()};
  }
};
inline constexpr as_expected_t as_expected{};

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

// --- The policy protocol: three optional capabilities discovered by introspection ----------
//
// Each probe follows the void_t partial-specialization idiom (see __never_returns_impl above):
// the primary reads false, the specialization is chosen only when the probed expression is
// well-formed. Probing never instantiates a member that is absent. A policy may arrive
// cv/ref-qualified, so each probe strips the reference and re-attaches an lvalue reference:
// policies are stored and invoked as non-const lvalues (retry mutates itself).

// Capability 1: the exception hook `p(const std::exception*, source_location)`.
template <class _AlwaysVoid, class _P>
inline constexpr bool __has_exception_hook_impl = false;

template <class _P>
inline constexpr bool __has_exception_hook_impl<
  ::cuda::std::void_t<decltype(::cuda::std::declval<_P&>()(
    ::cuda::std::declval<const ::std::exception*>(), ::cuda::std::declval<::cuda::std::source_location>()))>,
  _P> = true;

template <class _P>
inline constexpr bool __has_exception_hook = __has_exception_hook_impl<void, ::cuda::std::remove_reference_t<_P>>;

// The exception hook's return type -- the policy's "answer" (only named when the hook exists).
template <class _P>
using __hook_answer_t = decltype(::cuda::std::declval<::cuda::std::remove_reference_t<_P>&>()(
  ::cuda::std::declval<const ::std::exception*>(), ::cuda::std::declval<::cuda::std::source_location>()));

// Capability 2a: the success hook `p.on_success(R&&)` for a given result type.
template <class _AlwaysVoid, class _P, class _R>
inline constexpr bool __has_on_success_with_impl = false;

template <class _P, class _R>
inline constexpr bool __has_on_success_with_impl<
  ::cuda::std::void_t<decltype(::cuda::std::declval<_P&>().on_success(::cuda::std::declval<_R>()))>,
  _P,
  _R> = true;

template <class _P, class _R>
inline constexpr bool __has_on_success_with = __has_on_success_with_impl<void, ::cuda::std::remove_reference_t<_P>, _R>;

// Capability 2b: the nullary success hook `p.on_success()` (the void-result channel).
template <class _AlwaysVoid, class _P>
inline constexpr bool __has_on_success_void_impl = false;

template <class _P>
inline constexpr bool
  __has_on_success_void_impl<::cuda::std::void_t<decltype(::cuda::std::declval<_P&>().on_success())>, _P> = true;

template <class _P>
inline constexpr bool __has_on_success_void = __has_on_success_void_impl<void, ::cuda::std::remove_reference_t<_P>>;

// Capability 3: the around hook `p.invoke(F&&)` (middleware wrapping the callable).
template <class _AlwaysVoid, class _P, class _F>
inline constexpr bool __has_invoke_impl = false;

template <class _P, class _F>
inline constexpr bool
  __has_invoke_impl<::cuda::std::void_t<decltype(::cuda::std::declval<_P&>().invoke(::cuda::std::declval<_F>()))>,
                    _P,
                    _F> = true;

template <class _P, class _F>
inline constexpr bool __has_invoke = __has_invoke_impl<void, ::cuda::std::remove_reference_t<_P>, _F>;

// A policy is anything exposing at least one capability; the around probe uses a throwaway
// signature good enough to spot the member.
template <class _P>
inline constexpr bool __has_any_capability =
  __has_exception_hook<_P> || __has_on_success_void<_P> || __has_invoke<_P, void (&)()>;

// Whether a type is one this header defines as an exception-sink policy, marked by the
// __exception_sink_tag member. This is what &/| require of at least one operand, so they do
// not hijack unrelated types.
template <class _AlwaysVoid, class _P>
inline constexpr bool __is_exception_sink_impl = false;

template <class _P>
inline constexpr bool __is_exception_sink_impl<::cuda::std::void_t<typename _P::__exception_sink_tag>, _P> = true;

template <class _P>
inline constexpr bool __is_exception_sink_v = __is_exception_sink_impl<void, ::cuda::std::remove_cvref_t<_P>>;

// Whether a reaction is `::std::ignore` (compared as the current code does).
template <class _P>
inline constexpr bool __is_ignore_v =
  ::cuda::std::is_same_v<const ::cuda::std::remove_cvref_t<_P>,
                         const ::cuda::std::remove_reference_t<decltype(::std::ignore)>>;

// Whether a policy's answer type is `nothing` -- it never returns from the exception path. The
// two-step form keeps `__hook_answer_t` from being named for a hookless policy: `&&` does not
// short-circuit template instantiation, so the answer is probed only in the `true` partial.
template <bool _HasHook, class _P>
inline constexpr bool __answers_nothing_impl = false;

template <class _P>
inline constexpr bool __answers_nothing_impl<true, _P> =
  ::cuda::std::is_same_v<::cuda::std::remove_cvref_t<__hook_answer_t<_P>>, nothing>;

template <class _P>
inline constexpr bool __answers_nothing = __answers_nothing_impl<__has_exception_hook<_P>, _P>;

// Whether a policy's exception path is nothrow -- the same computation that
// `operator<<`'s conditional noexcept uses. Declining from `|` means throwing
// from that path, so a nothrow left side of `|` leaves the right unreachable.
template <class _P>
inline constexpr bool __exception_path_nothrow_v =
  __has_exception_hook<_P>
  && ::cuda::std::
    is_nothrow_invocable_v<::cuda::std::remove_reference_t<_P>&, const ::std::exception*, ::cuda::std::source_location>;

// --- Adapters: normalize the historical reactions into policies ----------------------------

// A nullary `nothing`-returning callable (bare `abort`, `terminate`, or a user's ending):
// report through `notify`, then run it. Its answer type `nothing` proves it never returns.
template <class _Action>
struct __terminating_action
{
  using __exception_sink_tag = void;
  _Action __action_;

  [[noreturn]] nothing operator()([[maybe_unused]] const ::std::exception* __exception,
                                  [[maybe_unused]] const ::cuda::std::source_location __loc) noexcept
  {
    notify(__exception, __loc);
    __action_();
    _CCCL_UNREACHABLE();
  }
};

// `::std::ignore` as a policy: resume with a default-constructed result.
struct __ignore_policy
{
  using __exception_sink_tag = void;

  decltype(::std::ignore) operator()([[maybe_unused]] const ::std::exception* __exception,
                                     [[maybe_unused]] const ::cuda::std::source_location __loc) const noexcept
  {
    return ::std::ignore;
  }
};

// `catch_only<E>(p)`: run `p`'s exception path only for an `E`, else decline by rethrowing.
template <class _E, class _P>
struct __catch_only_t
{
  using __exception_sink_tag = void;
  _P __p_;

  decltype(auto) operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc)
  {
    if (__exception != nullptr && dynamic_cast<const _E*>(__exception) != nullptr)
    {
      return __p_(__exception, __loc);
    }
    throw; // decline: wrong type, or a non-std exception (nullptr)
  }

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

  template <class _Fn, ::cuda::std::enable_if_t<__has_invoke<_P, _Fn>, int> = 0>
  decltype(auto) invoke(_Fn&& __fn)
  {
    return __p_.invoke(::cuda::std::forward<_Fn>(__fn));
  }
};

// Normalize any reaction into a policy object, returned by value (first match wins). A
// value stored inside carries its own ref-ness (e.g. subst_t<int&> for a reference result).
template <class _R>
auto __normalize(_R&& __r)
{
  using _P = ::cuda::std::remove_cvref_t<_R>;
  if constexpr (__has_any_capability<_P> || __is_exception_sink_v<_P>)
  {
    // Already a policy (has a capability, or a capability-free tagged type such as noop).
    return static_cast<_P>(::cuda::std::forward<_R>(__r));
  }
  else if constexpr (__never_returns<_P&>)
  {
    return __terminating_action<_R>{::cuda::std::forward<_R>(__r)};
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

// The type `__normalize` would produce for `_R`, and whether that type is a sink-tagged
// policy. Identity elimination for `&`/`|` consults this: only a tagged survivor is
// returned bare; an untagged one (raw effect lambda) must stay inside a composite so
// the chain remains composable.
template <class _R>
using __normalized_t = decltype(__normalize(::cuda::std::declval<_R>()));

template <class _R>
inline constexpr bool __normalizes_to_exception_sink_v = __is_exception_sink_v<__normalized_t<_R>>;

// Success- and around-hook forwarding shared by the `&` and `|` composites: the success hook
// is rightmost-wins (the right element owns the expression type), the around hook nests with
// the leftmost outermost. The exception hook -- where the two composites differ -- lives in
// the derived types.
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

  template <class _Fn, ::cuda::std::enable_if_t<__has_invoke<_L, _Fn> || __has_invoke<_R, _Fn>, int> = 0>
  decltype(auto) invoke(_Fn&& __fn)
  {
    if constexpr (__has_invoke<_L, _Fn> && __has_invoke<_R, _Fn>)
    {
      return __l_.invoke([&]() -> decltype(auto) {
        return __r_.invoke(::cuda::std::forward<_Fn>(__fn));
      });
    }
    else if constexpr (__has_invoke<_L, _Fn>)
    {
      return __l_.invoke(::cuda::std::forward<_Fn>(__fn));
    }
    else
    {
      return __r_.invoke(::cuda::std::forward<_Fn>(__fn));
    }
  }
};

// The sequencing composite `_L & _R`: on the exception path run `_L` then `_R`; `_R` answers.
template <class _L, class _R>
struct __policy_and : __composite_hooks<_L, _R>
{
  using __exception_sink_tag = void;

  static_assert(!__answers_nothing<_L>, "policies after a never-returning policy are unreachable");

  // Exception hook present iff either side has one. `_L` fires (answer discarded), then `_R`
  // answers; with no `_R` hook the composite's answer is `void`, which `__interpret_answer`
  // rejects in final position -- correct, since such a chain cannot answer on its own. The
  // `_LL`/`_RR` defaulted parameters make the constraint depend on this template, so it
  // SFINAEs away instead of hard-erroring at class instantiation.
  template <class _LL                                                                             = _L,
            class _RR                                                                             = _R,
            ::cuda::std::enable_if_t<__has_exception_hook<_LL> || __has_exception_hook<_RR>, int> = 0>
  decltype(auto) operator()([[maybe_unused]] const ::std::exception* __exception,
                            [[maybe_unused]] const ::cuda::std::source_location __loc)
  {
    if constexpr (__has_exception_hook<_L>)
    {
      static_cast<void>(this->__l_(__exception, __loc));
      if constexpr (__has_exception_hook<_R>)
      {
        return this->__r_(__exception, __loc);
      }
    }
    else
    {
      return this->__r_(__exception, __loc);
    }
  }
};

template <class _Ln, class _Rn>
auto __make_and(_Ln&& __l, _Rn&& __r)
{
  return __policy_and<_Ln, _Rn>{{::cuda::std::forward<_Ln>(__l), ::cuda::std::forward<_Rn>(__r)}};
}

// The alternation composite `_L | _R`: `_L` claims first; if it declines by throwing, `_R`
// handles the original (re-observed) exception.
template <class _L, class _R>
struct __policy_or : __composite_hooks<_L, _R>
{
  using __exception_sink_tag = void;

  static_assert(__has_exception_hook<_L> && __has_exception_hook<_R>,
                "both sides of | must answer the exception path (have an exception hook)");
  static_assert(!__exception_path_nothrow_v<_L>,
                "the left policy never declines; alternatives after it are unreachable");

  // The answer is `_R`'s if `_L`'s is `nothing`, else `_L`'s; the answer-to-expression
  // conversion in `__interpret_answer` is where any mismatch surfaces.
  using __answer = ::cuda::std::conditional_t<__answers_nothing<_L>, __hook_answer_t<_R>, __hook_answer_t<_L>>;

  __answer operator()(const ::std::exception* __exception, const ::cuda::std::source_location __loc)
  {
    _CCCL_TRY
    {
      if constexpr (__answers_nothing<_L>)
      {
        this->__l_(__exception, __loc);
        _CCCL_UNREACHABLE();
      }
      else
      {
        return this->__l_(__exception, __loc);
      }
    }
    _CCCL_CATCH_ALL
    {
      // Re-observe the still-active exception so `_R` gets a correct pointer, mirroring the
      // catch pair in `operator<<` even when `_L` declined a non-std exception.
      _CCCL_TRY
      {
        throw;
      }
      _CCCL_CATCH (const ::std::exception& __e)
      {
        return this->__r_(&__e, __loc);
      }
      _CCCL_CATCH_ALL
      {
        return this->__r_(nullptr, __loc);
      }
    }
  }
};

template <class _Ln, class _Rn>
auto __make_or(_Ln&& __l, _Rn&& __r)
{
  return __policy_or<_Ln, _Rn>{{::cuda::std::forward<_Ln>(__l), ::cuda::std::forward<_Rn>(__r)}};
}

template <class _E, class _Normalized>
auto __make_catch_only(_Normalized&& __p)
{
  return __catch_only_t<_E, _Normalized>{::cuda::std::forward<_Normalized>(__p)};
}

// Interpret the final element's answer as the expression's value, converting to `_Expr`.
template <class _Expr, class _P>
_Expr __interpret_answer(_P& __policy, const ::std::exception* __exception, const ::cuda::std::source_location __loc)
{
  using _Answer = __hook_answer_t<_P>;
  static_assert(!::cuda::std::is_void_v<_Answer>,
                "the final policy must answer the exception path: nothing to die, ::std::ignore "
                "to resume, or a value to substitute");

  if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Answer>, nothing>)
  {
    // Never returns: no backstop beyond the unreachable marker.
    __policy(__exception, __loc);
    _CCCL_UNREACHABLE();
  }
  else if constexpr (__is_ignore_v<_Answer>)
  {
    // Resume: default-construct the expression's value (nothing to do for void).
    static_cast<void>(__policy(__exception, __loc));
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
    return static_cast<_Expr>(__policy(__exception, __loc));
  }
}

// Walk the chain on the exception path: with no hook anywhere, propagate; else interpret.
// The parameters go unread in the propagate instantiation, which gcc 9 flags without the
// attribute.
template <class _Expr, class _P>
_Expr __on_exception(_P& __policy,
                     [[maybe_unused]] const ::std::exception* __exception,
                     [[maybe_unused]] const ::cuda::std::source_location __loc)
{
  if constexpr (!__has_exception_hook<_P>)
  {
    throw; // no element answered (e.g. on_throw(retry(3))): let the exception propagate
  }
  else
  {
    return __interpret_answer<_Expr>(__policy, __exception, __loc);
  }
}

// The policy carrier. ADL finds `operator<<` here since the type lives in this namespace.
template <class _Reaction>
struct __on_throw_policy
{
  _Reaction __reaction_;
  const ::cuda::std::source_location __loc_;
};

template <class _Reaction>
auto __make_on_throw(_Reaction&& __reaction, const ::cuda::std::source_location __loc)
{
  return __on_throw_policy<_Reaction>{::cuda::std::forward<_Reaction>(__reaction), __loc};
}

template <class _Reaction, class _Fn>
// A resuming chain reads neither exception nor location in some instantiations; gcc 9 flags the
// unread policy without the attribute.
decltype(auto) operator<<([[maybe_unused]] __on_throw_policy<_Reaction> __policy,
                          _Fn&& __fn) noexcept(__exception_path_nothrow_v<_Reaction>)
{
  // A `noexcept` callable puts the policy out of reach: an exception raised inside it ends the
  // program where it stands, so the catch below could never run and the policy would be a
  // promise nobody keeps.
  static_assert(!noexcept(::cuda::std::forward<_Fn>(__fn)()),
                "on_throw has nothing to do for a noexcept callable, which terminates rather than "
                "throws; call such a callable directly");

  using _Result = decltype(::cuda::std::forward<_Fn>(__fn)());
  using _P      = _Reaction;

  if constexpr (::cuda::std::is_void_v<_Result>)
  {
    if constexpr (__has_on_success_void<_P>)
    {
      // A success hook owns the type (e.g. defer, as_expected): the expression is what it makes.
      using _Expr = decltype(::cuda::std::declval<_P&>().on_success());
      _CCCL_TRY
      {
        if constexpr (__has_invoke<_P, _Fn>)
        {
          __policy.__reaction_.invoke(::cuda::std::forward<_Fn>(__fn));
        }
        else
        {
          ::cuda::std::forward<_Fn>(__fn)();
        }
        return __policy.__reaction_.on_success();
      }
      _CCCL_CATCH (const ::std::exception& __exception)
      {
        return detail::__on_exception<_Expr>(__policy.__reaction_, &__exception, __policy.__loc_);
      }
      _CCCL_CATCH_ALL
      {
        return detail::__on_exception<_Expr>(__policy.__reaction_, nullptr, __policy.__loc_);
      }
    }
    else
    {
      // No success hook: the expression is void, the result passing through.
      _CCCL_TRY
      {
        if constexpr (__has_invoke<_P, _Fn>)
        {
          __policy.__reaction_.invoke(::cuda::std::forward<_Fn>(__fn));
        }
        else
        {
          ::cuda::std::forward<_Fn>(__fn)();
        }
      }
      _CCCL_CATCH (const ::std::exception& __exception)
      {
        return detail::__on_exception<void>(__policy.__reaction_, &__exception, __policy.__loc_);
      }
      _CCCL_CATCH_ALL
      {
        return detail::__on_exception<void>(__policy.__reaction_, nullptr, __policy.__loc_);
      }
    }
  }
  else if constexpr (__has_on_success_with<_P, _Result>)
  {
    // A success hook transforms/owns the non-void result; the expression is its return type.
    using _Expr = decltype(::cuda::std::declval<_P&>().on_success(::cuda::std::declval<_Result>()));
    _CCCL_TRY
    {
      if constexpr (__has_invoke<_P, _Fn>)
      {
        return __policy.__reaction_.on_success(__policy.__reaction_.invoke(::cuda::std::forward<_Fn>(__fn)));
      }
      else
      {
        return __policy.__reaction_.on_success(::cuda::std::forward<_Fn>(__fn)());
      }
    }
    _CCCL_CATCH (const ::std::exception& __exception)
    {
      return detail::__on_exception<_Expr>(__policy.__reaction_, &__exception, __policy.__loc_);
    }
    _CCCL_CATCH_ALL
    {
      return detail::__on_exception<_Expr>(__policy.__reaction_, nullptr, __policy.__loc_);
    }
  }
  else
  {
    // No hook accepts the non-void result: pass it through, unless a void-only success hook
    // (e.g. defer over a non-void callable) means the result has no channel.
    static_assert(!__has_on_success_void<_P>, "the policy's on_success cannot accept the callable's result");
    _CCCL_TRY
    {
      if constexpr (__has_invoke<_P, _Fn>)
      {
        return __policy.__reaction_.invoke(::cuda::std::forward<_Fn>(__fn));
      }
      else
      {
        return ::cuda::std::forward<_Fn>(__fn)();
      }
    }
    _CCCL_CATCH (const ::std::exception& __exception)
    {
      return detail::__on_exception<_Result>(__policy.__reaction_, &__exception, __policy.__loc_);
    }
    _CCCL_CATCH_ALL
    {
      return detail::__on_exception<_Result>(__policy.__reaction_, nullptr, __policy.__loc_);
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
 * A policy is an object exposing any of three optional capabilities, discovered by compile-time
 * introspection: an exception hook `(const std::exception*, source_location)` whose return
 * value is its answer on the throw path, a success hook `on_success(...)` that observes or
 * replaces the result, and an around hook `invoke(callable)` that wraps the call itself. The
 * named policies are @ref notify_t "notify", @ref subst_t "subst", @ref defer_t "defer",
 * @ref rethrow_t "rethrow", @ref retry_t "retry", @ref as_expected_t "as_expected",
 * @ref noop_t "noop", and @ref catch_only. Policies compose with `&` (sequence; the last
 * element answers) and `|` (alternation; the left may decline by throwing).
 *
 * For backward compatibility `on_throw` also accepts non-policy reactions: `abort` and
 * `terminate` (nullary `nothing`-returning functions) report through `notify` and then run;
 * `std::ignore` resumes with a default-constructed result; and anything else is taken as a
 * substitution value, exactly as `subst(value)`. A substitution passed as an lvalue can serve
 * a reference result, which the policy refers to rather than copies:
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
  return detail::__make_on_throw(detail::__normalize(::cuda::std::forward<_Reaction>(__reaction)), __loc);
}

/**
 * @brief Restricts a policy to exceptions deriving from `E`: `catch_only<E>(p)` runs `p`'s
 * exception path when the caught exception is an `E`, and otherwise declines by rethrowing --
 * reconstructing a `catch (const E&)` clause for use as the left arm of a `|` ladder. A
 * non-`std::exception` (which reaches the handler as `nullptr`) always declines. `E` must
 * derive from `std::exception`.
 */
template <class _E, class _P>
auto catch_only(_P&& __p)
{
  static_assert(::cuda::std::is_base_of_v<::std::exception, _E>,
                "catch_only<E> requires E to derive from std::exception");
  return detail::__make_catch_only<_E>(detail::__normalize(::cuda::std::forward<_P>(__p)));
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
  return detail::__make_and(detail::__normalize(::cuda::std::forward<_L>(__l)),
                            detail::__normalize(::cuda::std::forward<_R>(__r)));
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
  return detail::__make_or(detail::__normalize(::cuda::std::forward<_L>(__l)),
                           detail::__normalize(::cuda::std::forward<_R>(__r)));
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
  const auto die = [](const ::std::exception*, ::cuda::std::source_location) noexcept -> nothing {
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
  const auto tick = [&hits](const ::std::exception*, ::cuda::std::source_location) noexcept {
    ++hits;
  };
  const int ticked = on_throw(noop & tick & ::std::ignore) << []() -> int {
    throw ::std::runtime_error("counted");
  };
  EXPECT(ticked == 0);
  EXPECT(hits == 1);

  // Reporting somewhere other than stderr: notify(stream) is a configured copy of notify.
  ::FILE* const log = ::tmpfile();
  EXPECT(log != nullptr);
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
  EXPECT(::fgets(message, sizeof(message), log) != nullptr);
  ::snprintf(expected,
             sizeof(expected),
             "%s(%u) on_throw violation in %s: boom\n",
             site.file_name(),
             site.line(),
             site.function_name());
  EXPECT(::std::string_view{message} == expected);
  EXPECT(::fgets(message, sizeof(message), log) != nullptr);
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
    return [&trace, c](const ::std::exception*, ::cuda::std::source_location) noexcept {
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
    on_throw(retry(2)) << [&] {
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

  // retry & terminal: the terminal handles the exhausted failure. Success stops the loop.
  attempts      = 0;
  const int r13 = on_throw(retry(5) & subst(-1)) << [&]() -> int {
    if (++attempts < 3)
    {
      throw ::std::runtime_error("transient");
    }
    return 99;
  };
  EXPECT(r13 == 99);
  EXPECT(attempts == 3);

  attempts      = 0;
  const int r14 = on_throw(retry(1) & subst(-1)) << [&]() -> int {
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

  // Elimination still normalizes: noop & abort is the terminating adapter, and works.
  {
    using cuda::experimental::stf::abort;
    const int kept = on_throw(noop & abort) << [] {
      return 11;
    };
    EXPECT(kept == 11);
  }

  // Behavior after elimination is unchanged (the r6/r7 identity tests above already
  // exercise the runtime side; keep them).

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

  const int s3 = on_throw(subst([](const ::std::exception* e, ::cuda::std::source_location) noexcept {
                   return e != nullptr ? 1 : 2;
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
