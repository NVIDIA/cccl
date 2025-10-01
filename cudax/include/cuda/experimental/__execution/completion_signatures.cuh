//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_COMPLETION_SIGNATURES
#define __CUDAX_EXECUTION_COMPLETION_SIGNATURES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

// include this last:
#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-but-set-parameter")

namespace cuda::experimental::execution
{
using ::cuda::std::__type_list;

// __partitioned_completions is a cache of completion signatures for fast
// access. The completion_signatures<Sigs...>::__partitioned nested struct
// inherits from __partitioned_completions. If the cache is never accessed,
// it is never instantiated.
template <class _ValueTuplesList = __type_list<>, class _ErrorsList = __type_list<>, class _StoppedList = __type_list<>>
struct __partitioned_completions;

template <class... _ValueTuples, class... _Errors, class... _Stopped>
struct __partitioned_completions<__type_list<_ValueTuples...>, __type_list<_Errors...>, __type_list<_Stopped...>>
{
  template <template <class...> class _Tuple, template <class...> class _Variant>
  using __value_types _CCCL_NODEBUG_ALIAS =
    _Variant<::cuda::std::__type_call1<_ValueTuples, ::cuda::std::__type_quote<_Tuple>>...>;

  template <template <class...> class _Variant, template <class...> class _Transform = ::cuda::std::__type_self_t>
  using __error_types _CCCL_NODEBUG_ALIAS = _Variant<_Transform<_Errors>...>;

  template <template <class...> class _Variant, class _Type = set_stopped_t()>
  using __stopped_types _CCCL_NODEBUG_ALIAS = _Variant<__type_second<_Stopped, _Type>...>;

  using __count_values  = ::cuda::std::integral_constant<size_t, sizeof...(_ValueTuples)>;
  using __count_errors  = ::cuda::std::integral_constant<size_t, sizeof...(_Errors)>;
  using __count_stopped = ::cuda::std::integral_constant<size_t, sizeof...(_Stopped)>;

  struct __nothrow_decay_copyable
  {
    // These aliases are placed in a separate struct to avoid computing them
    // if they are not needed.
    using __fn     = ::cuda::std::__type_quote<__nothrow_decay_copyable_t>;
    using __values = ::cuda::std::_And<::cuda::std::__type_call1<_ValueTuples, __fn>...>;
    using __errors = __nothrow_decay_copyable_t<_Errors...>;
    using __all    = ::cuda::std::_And<__values, __errors>;
  };
};

template <class _Tag>
struct __partitioned_fold_fn;

template <>
struct __partitioned_fold_fn<set_value_t>
{
  template <class... _ValueTuples, class _Errors, class _Stopped, class _Values>
  _CCCL_API auto operator()(__partitioned_completions<__type_list<_ValueTuples...>, _Errors, _Stopped>&,
                            ::cuda::std::__undefined<_Values>&) const
    -> ::cuda::std::__undefined<__partitioned_completions<__type_list<_ValueTuples..., _Values>, _Errors, _Stopped>>&;
};

template <>
struct __partitioned_fold_fn<set_error_t>
{
  template <class _Values, class... _Errors, class _Stopped, class _Error>
  _CCCL_API auto operator()(__partitioned_completions<_Values, __type_list<_Errors...>, _Stopped>&,
                            ::cuda::std::__undefined<__type_list<_Error>>&) const
    -> ::cuda::std::__undefined<__partitioned_completions<_Values, __type_list<_Errors..., _Error>, _Stopped>>&;
};

template <>
struct __partitioned_fold_fn<set_stopped_t>
{
  template <class _Values, class _Errors, class _Stopped>
  _CCCL_API auto operator()(__partitioned_completions<_Values, _Errors, _Stopped>&, ::cuda::std::__ignore_t) const
    -> ::cuda::std::__undefined<__partitioned_completions<_Values, _Errors, __type_list<set_stopped_t()>>>&;
};

// The following overload of binary operator* is used to build up the cache of completion
// signatures. We fold over operator*, accumulating the completion signatures in the
// cache. `__undefined` is used here to prevent the instantiation of the intermediate
// types.
template <class _Partitioned, class _Tag, class... _Args>
_CCCL_API auto operator*(::cuda::std::__undefined<_Partitioned>&, _Tag (*)(_Args...)) -> ::cuda::std::
  __call_result_t<__partitioned_fold_fn<_Tag>, _Partitioned&, ::cuda::std::__undefined<__type_list<_Args...>>&>;

// This function declaration is used to extract the cache from the `__undefined` type.
template <class _Partitioned>
_CCCL_API auto __unpack_partitioned_completions(::cuda::std::__undefined<_Partitioned>&) -> _Partitioned;

template <class... _Sigs>
using __partition_completion_signatures_t _CCCL_NODEBUG_ALIAS = //
  decltype(execution::__unpack_partitioned_completions(
    (declval<::cuda::std::__undefined<__partitioned_completions<>>&>() * ... * static_cast<_Sigs*>(nullptr))));

template <class _Completions>
using __partitioned_completions_of _CCCL_NODEBUG_ALIAS = typename _Completions::__partitioned::type;

////////////////////////////////////////////////////////////////////////////////////////////////////
// completion signatures type traits
template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
using __value_types _CCCL_NODEBUG_ALIAS =
  typename __partitioned_completions_of<_Sigs>::template __value_types<_Tuple, _Variant>;

template <class _Sndr, class _Env, template <class...> class _Tuple, template <class...> class _Variant>
using value_types_of_t _CCCL_NODEBUG_ALIAS =
  __value_types<completion_signatures_of_t<_Sndr, _Env>,
                ::cuda::std::__type_indirect_quote<_Tuple>::template __call,
                ::cuda::std::__type_indirect_quote<_Variant>::template __call>;

template <class _Sigs,
          template <class...> class _Variant,
          template <class...> class _Transform = ::cuda::std::__type_self_t>
using __error_types _CCCL_NODEBUG_ALIAS =
  typename __partitioned_completions_of<_Sigs>::template __error_types<_Variant, _Transform>;

template <class _Sndr, class _Env, template <class...> class _Variant>
using error_types_of_t _CCCL_NODEBUG_ALIAS =
  __error_types<completion_signatures_of_t<_Sndr, _Env>, ::cuda::std::__type_indirect_quote<_Variant>::template __call>;

template <class _Sigs, template <class...> class _Variant, class _Type = set_stopped_t()>
using __stopped_types _CCCL_NODEBUG_ALIAS =
  typename __partitioned_completions_of<_Sigs>::template __stopped_types<_Variant, _Type>;

template <class _Sigs>
inline constexpr bool __sends_stopped = __partitioned_completions_of<_Sigs>::__count_stopped::value != 0;

template <class _Sndr, class... _Env>
inline constexpr bool sends_stopped = __sends_stopped<completion_signatures_of_t<_Sndr, _Env...>>;

////////////////////////////////////////////////////////////////////////////////////////////////////
// __valid_completion_signatures
template <class _Ty>
_CCCL_CONCEPT __valid_completion_signatures =
  ::cuda::__is_specialization_of_v<::cuda::std::remove_const_t<_Ty>, completion_signatures>;

template <class... _Sigs>
_CCCL_API _CCCL_CONSTEVAL void __assert_valid_completion_signatures(const completion_signatures<_Sigs...>&)
{}

////////////////////////////////////////////////////////////////////////////////////////////////////
// make_completion_signatures
template <class _Tag, class... _As>
_CCCL_API auto __normalize_impl(_As&&...) -> _Tag (*)(_As...);

template <class _Tag, class... _As>
_CCCL_API auto __normalize(_Tag (*)(_As...)) -> decltype(execution::__normalize_impl<_Tag>(declval<_As>()...));

template <class... _Sigs>
_CCCL_API auto __make_unique(_Sigs*...)
  -> ::cuda::std::__type_apply<::cuda::std::__type_quote<completion_signatures>, ::cuda::std::__make_type_set<_Sigs...>>;

template <class... _Sigs>
using __make_completion_signatures_t _CCCL_NODEBUG_ALIAS =
  decltype(execution::__make_unique(execution::__normalize(static_cast<_Sigs*>(nullptr))...));

template <class... _ExplicitSigs, class... _DeducedSigs>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto make_completion_signatures(_DeducedSigs*...) noexcept
  -> __make_completion_signatures_t<_ExplicitSigs..., _DeducedSigs...>
{
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// concat_completion_signatures
struct __concat_completion_signatures_impl;

template <class... _Sigs>
using __concat_completion_signatures_t _CCCL_NODEBUG_ALIAS =
  __call_result_t<__call_result_t<__concat_completion_signatures_impl, const _Sigs&...>>;

struct __concat_completion_signatures_fn
{
  template <class... _Sigs>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()(const _Sigs&...) const noexcept
    -> __concat_completion_signatures_t<_Sigs...>
  {
    return {};
  }
};

extern const completion_signatures<>& __empty_completion_signatures;

struct __concat_completion_signatures_impl
{
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()() const noexcept -> completion_signatures<> (*)()
  {
    return nullptr;
  }

  template <class... _Sigs>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()(const completion_signatures<_Sigs...>&) const noexcept
    -> __make_completion_signatures_t<_Sigs...> (*)()
  {
    return nullptr;
  }

  template <class _Self = __concat_completion_signatures_impl,
            class... _As,
            class... _Bs,
            class... _Cs,
            class... _Ds,
            class... _Rest>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()(
    const completion_signatures<_As...>&,
    const completion_signatures<_Bs...>&,
    const completion_signatures<_Cs...>& = __empty_completion_signatures,
    const completion_signatures<_Ds...>& = __empty_completion_signatures,
    const _Rest&...) const noexcept
  {
    using _Tmp                           = completion_signatures<_As..., _Bs..., _Cs..., _Ds...>;
    using _SigsFnPtr _CCCL_NODEBUG_ALIAS = __call_result_t<_Self, const _Tmp&, const _Rest&...>;
    return static_cast<_SigsFnPtr>(nullptr);
  }

  template <class _Ap,
            class _Bp = ::cuda::std::__ignore_t,
            class _Cp = ::cuda::std::__ignore_t,
            class _Dp = ::cuda::std::__ignore_t,
            class... _Rest>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto
  operator()(const _Ap&, const _Bp& = {}, const _Cp& = {}, const _Dp& = {}, const _Rest&...) const noexcept
  {
    if constexpr (!__valid_completion_signatures<_Ap>)
    {
      return static_cast<_Ap (*)()>(nullptr);
    }
    else if constexpr (!__valid_completion_signatures<_Bp>)
    {
      return static_cast<_Bp (*)()>(nullptr);
    }
    else if constexpr (!__valid_completion_signatures<_Cp>)
    {
      return static_cast<_Cp (*)()>(nullptr);
    }
    else
    {
      static_assert(!__valid_completion_signatures<_Dp>);
      return static_cast<_Dp (*)()>(nullptr);
    }
  }
};

_CCCL_GLOBAL_CONSTANT __concat_completion_signatures_fn concat_completion_signatures{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// implementation details of the completion_signatures class template
struct _IN_COMPLETION_SIGNATURES_APPLY;
struct _IN_COMPLETION_SIGNATURES_TRANSFORM_REDUCE;
struct _FUNCTION_IS_NOT_CALLABLE_WITH_THESE_SIGNATURES;

template <class... _Sigs>
struct __remove_sigs
{
  template <class _Sig>
  _CCCL_API constexpr auto operator()(_Sig*) const noexcept -> bool
  {
    return !::cuda::std::__type_set_contains_v<::cuda::std::__type_set<_Sigs...>, _Sig>;
  }
};

template <class _Fn, class _Sig>
_CCCL_API _CCCL_CONSTEVAL auto __filer_one() noexcept
  -> ::cuda::std::_If<_Fn{}(static_cast<_Sig*>(nullptr)), completion_signatures<_Sig>, completion_signatures<>>
{
  return {};
}

// working around compiler bugs in gcc and msvc
template <class... _Sigs>
using __completion_signatures = completion_signatures<_Sigs...>;

template <class... _Values>
using __set_value_sig_t = set_value_t(_Values...);

template <class _Error>
using __set_error_sig_t = set_error_t(_Error);

//! @brief Represents a set of completion signatures for senders in the CUDA C++ execution
//! model.
//!
//! The `completion_signatures` class template is used to describe the possible ways a
//! sender may complete. Each signature is a function type of the form
//! `set_value_t(Ts...)`, `set_error_t(E)`, or `set_stopped_t()`. This type provides
//! compile-time utilities for querying, combining, and transforming sets of completion
//! signatures.
//!
//! @tparam _Sigs... The completion signature types to include in this set.
//!
//! Example usage:
//! @code
//! constexpr auto sigs = completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};
//! static_assert(sigs.size() == 3);
//! static_assert(sigs.contains<set_value_t(int)>());
//! @endcode
template <class... _Sigs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT completion_signatures
{
  //! @brief Partitioned view of the completion signatures for efficient querying.
  struct __partitioned
  {
    // This is defined in a nested struct to avoid computing these types if they are not
    // needed.
    using type _CCCL_NODEBUG_ALIAS = __partition_completion_signatures_t<_Sigs...>;
  };

  //! @brief Type set view of the completion signatures for set operations.
  struct __type_set
  {
    // This is defined in a nested struct to avoid computing this type if it is not
    // needed.
    using type _CCCL_NODEBUG_ALIAS = ::cuda::std::__make_type_set<_Sigs...>;
  };

  //! @brief Applies a metafunction to each signature and collects the results.
  //! @tparam _Fn The metafunction to apply.
  //! @tparam _Continuation The template to collect results into.
  template <template <class...> class _Fn, template <class...> class _Continuation = __completion_signatures>
  using __transform_q _CCCL_NODEBUG_ALIAS = _Continuation<::cuda::std::__type_apply_q<_Fn, _Sigs>...>;

  //! @brief Applies a callable metafunction to each signature and collects the results.
  //! @tparam _Fn The callable metafunction to apply.
  //! @tparam _Continuation The template to collect results into.
  template <class _Fn, class _Continuation = ::cuda::std::__type_quote<__completion_signatures>>
  using __transform _CCCL_NODEBUG_ALIAS = __transform_q<_Fn::template __call, _Continuation::template __call>;

  //! @brief Calls a metafunction with the signatures as arguments.
  //! @tparam _Fn The metafunction to call.
  //! @tparam _More Additional arguments to pass.
  template <class _Fn, class... _More>
  using __call _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_call<_Fn, _More..., _Sigs...>;

  //! @brief Default constructor.
  _CCCL_HIDE_FROM_ABI constexpr completion_signatures() = default;

  //! @brief Returns the number of completion signatures in the set.
  //! @return The number of signatures.
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto size() noexcept -> size_t
  {
    return sizeof...(_Sigs);
  }

  //! @brief Counts the number of signatures with the given tag.
  //! @tparam _Tag The tag to count (e.g., set_value, set_error, set_stopped).
  //! @return The number of signatures with the given tag.
  template <class _Tag>
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto count(_Tag) noexcept -> size_t
  {
    if constexpr (_Tag{} == set_value)
    {
      return __partitioned::type::__count_values::value;
    }
    else if constexpr (_Tag{} == set_error)
    {
      return __partitioned::type::__count_errors::value;
    }
    else
    {
      return __partitioned::type::__count_stopped::value;
    }
  }

  //! @brief Checks if the set contains the given signature.
  //! @tparam _Sig The signature type to check.
  //! @return true if the signature is present, false otherwise.
  template <class _Sig>
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto contains(_Sig* = nullptr) noexcept -> bool
  {
    return ::cuda::std::__type_set_contains_v<typename __type_set::type, _Sig>;
  }

  //! @brief Applies a callable to all signatures in the set.
  //! @tparam _Fn The callable to apply.
  //! @param __fn The callable instance.
  //! @return The result of calling __fn with all signatures as arguments.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn>
  _CCCL_API static _CCCL_CONSTEVAL auto apply(_Fn __fn) -> __call_result_t<_Fn, _Sigs*...>
  {
    return __fn(static_cast<_Sigs*>(nullptr)...);
  }

  //! @brief Filters the set using a predicate, returning a new set with only matching
  //! signatures.
  //! @tparam _Fn The predicate type (must be empty and trivially constructible).
  //! @param The predicate instance.
  //! @return A new completion_signatures set with only the signatures for which the
  //! predicate returns true.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn>
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto filter(_Fn)
  {
    static_assert(::cuda::std::is_empty_v<_Fn> && ::cuda::std::is_trivially_constructible_v<_Fn>,
                  "The filter function must be empty and trivially constructible.");
    return concat_completion_signatures(execution::__filer_one<_Fn, _Sigs>()...);
  }

  //! @brief Selects all signatures with the given tag.
  //! @tparam _Tag The tag to select (e.g., set_value, set_error, set_stopped).
  //! @return A new completion_signatures set containing only signatures with the given
  //! tag.
  template <class _Tag>
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto select(_Tag) noexcept
  {
    if constexpr (_Tag{} == set_value)
    {
      return __value_types<completion_signatures, __set_value_sig_t, __completion_signatures>{};
    }
    else if constexpr (_Tag{} == set_error)
    {
      return __error_types<completion_signatures, __completion_signatures, __set_error_sig_t>{};
    }
    else
    {
      static_assert(_Tag{} == set_stopped, "The tag must be set_value, set_error, or set_stopped.");
      return __stopped_types<completion_signatures, __completion_signatures>{};
    }
  }

  //! @brief Applies a transform and then reduces the results.
  //! @tparam _Transform The transform callable.
  //! @tparam _Reduce The reduce callable.
  //! @param __transform The transform instance.
  //! @param __reduce The reduce instance.
  //! @return The result of reducing the transformed signatures.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Transform, class _Reduce>
  [[nodiscard]]
  _CCCL_API static _CCCL_CONSTEVAL auto transform_reduce(_Transform __transform, _Reduce __reduce)
    -> __call_result_t<_Reduce, __call_result_t<_Transform, _Sigs*>...>
  {
    return __reduce(__transform(static_cast<_Sigs*>(nullptr))...);
  }
};

_CCCL_HOST_DEVICE completion_signatures() -> completion_signatures<>;

// work-around for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95629
#if _CCCL_COMPILER(GCC, ==, 11)
#  define _CCCL_CONSTEVAL_OPERATOR constexpr
#else // ^^^ GCC 11 ^^^ / vvv other compilers vvv
#  define _CCCL_CONSTEVAL_OPERATOR _CCCL_CONSTEVAL
#endif // ^^^ other compilers ^^^

//! @brief Returns the union of two sets of completion signatures.
//! @tparam _SelfSigs The first set of signature types.
//! @tparam _OtherSigs The other set of signature types.
//! @param __self The first `completion_signatures` object.
//! @param __other The other `completion_signatures` object.
//! @return The union of the two sets.
template <class... _SelfSigs, class... _OtherSigs>
[[nodiscard]]
_CCCL_API _CCCL_CONSTEVAL_OPERATOR auto
operator+([[maybe_unused]] completion_signatures<_SelfSigs...> __self,
          [[maybe_unused]] completion_signatures<_OtherSigs...> __other) noexcept
{
  if constexpr (sizeof...(_SelfSigs) == 0) // short-circuit some common cases
  {
    return __other;
  }
  else if constexpr (sizeof...(_OtherSigs) == 0)
  {
    return __self;
  }
  else
  {
    return concat_completion_signatures(__self, __other);
  }
}

//! @brief Returns the set difference between two sets of completion signatures.
//! @tparam _SelfSigs The first set of signature types.
//! @tparam _OtherSigs The second set of signature types.
//! @return A new set with all signatures from the other set removed.
template <class... _SelfSigs, class... _OtherSigs>
[[nodiscard]]
_CCCL_API _CCCL_CONSTEVAL_OPERATOR auto
operator-(completion_signatures<_SelfSigs...> __self, completion_signatures<_OtherSigs...>) noexcept
{
  if constexpr (sizeof...(_OtherSigs) == 0 || sizeof...(_SelfSigs) == 0) // short-circuit some common cases
  {
    return __self;
  }
  else
  {
    return __self.filter(__remove_sigs<_OtherSigs...>{});
  }
}

//! @brief Checks if two completion_signatures sets are equal.
//! @tparam _SelfSigs The first set of signature types.
//! @tparam _OtherSigs The second set of signature types.
//! @return `true` if the sets are equal, `false` otherwise.
template <class... _SelfSigs, class... _OtherSigs>
[[nodiscard]]
_CCCL_API _CCCL_CONSTEVAL_OPERATOR auto
operator==(completion_signatures<_SelfSigs...>, completion_signatures<_OtherSigs...>) noexcept -> bool
{
  if constexpr (sizeof...(_OtherSigs) != sizeof...(_SelfSigs))
  {
    return false;
  }
  else
  {
    using __signatures_set_t = typename completion_signatures<_SelfSigs...>::__type_set::type;
    return ::cuda::std::__type_set_contains_v<__signatures_set_t, _OtherSigs...>;
  }
}

//! @brief Checks if two completion_signatures sets are not equal.
//! @tparam _SelfSigs The first set of signature types.
//! @tparam _OtherSigs The second set of signature types.
//! @param __self The other `completion_signatures` object.
//! @param __other The other `completion_signatures` object.
//! @return `true` if the sets are not equal, `false` otherwise.
template <class... _SelfSigs, class... _OtherSigs>
[[nodiscard]]
_CCCL_API _CCCL_CONSTEVAL_OPERATOR auto
operator!=(completion_signatures<_SelfSigs...> __self, completion_signatures<_OtherSigs...> __other) noexcept -> bool
{
  return !(__self == __other);
}

#undef _CCCL_CONSTEVAL_OPERATOR

////////////////////////////////////////////////////////////////////////////////////////////////////
// __gather_completion_signatures
template <class _WantedTag>
struct __gather_sigs_fn;

template <>
struct __gather_sigs_fn<set_value_t>
{
  template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
  using __call _CCCL_NODEBUG_ALIAS = __value_types<_Sigs, _Tuple, _Variant>;
};

template <>
struct __gather_sigs_fn<set_error_t>
{
  template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
  using __call _CCCL_NODEBUG_ALIAS = __error_types<_Sigs, _Variant, _Tuple>;
};

template <>
struct __gather_sigs_fn<set_stopped_t>
{
  template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
  using __call _CCCL_NODEBUG_ALIAS = __stopped_types<_Sigs, _Variant, _Tuple<>>;
};

template <class _Sigs, class _WantedTag, template <class...> class _Tuple, template <class...> class _Variant>
using __gather_completion_signatures _CCCL_NODEBUG_ALIAS =
  typename __gather_sigs_fn<_WantedTag>::template __call<_Sigs, _Tuple, _Variant>;

////////////////////////////////////////////////////////////////////////////////////////////////////
// __eptr_completion and __eptr_completion_if
#if _CCCL_HAS_EXCEPTIONS()
[[nodiscard]] _CCCL_API inline _CCCL_CONSTEVAL auto __eptr_completion() noexcept
{
  return completion_signatures<set_error_t(::std::exception_ptr)>{};
}
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
[[nodiscard]] _CCCL_API inline _CCCL_CONSTEVAL auto __eptr_completion() noexcept
{
  return completion_signatures{};
}
#endif // !_CCCL_HAS_EXCEPTIONS()

template <bool _PotentiallyThrowing>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __eptr_completion_if() noexcept
{
  if constexpr (_PotentiallyThrowing)
  {
    return __eptr_completion();
  }
  else
  {
    return completion_signatures{};
  }
}

using __eptr_completion_t _CCCL_NODEBUG_ALIAS = decltype(execution::__eptr_completion());

template <bool _PotentiallyThrowing>
using __eptr_completion_if_t _CCCL_NODEBUG_ALIAS = decltype(execution::__eptr_completion_if<_PotentiallyThrowing>());

////////////////////////////////////////////////////////////////////////////////////////////////////
// invalid_completion_signature
#if _CCCL_HAS_EXCEPTIONS() && __cpp_constexpr_exceptions >= 202411L // C++26, https://wg21.link/p3068

template <class... _What, class... _Values>
[[noreturn, nodiscard]] _CCCL_API consteval auto invalid_completion_signature(_Values... __values)
  -> completion_signatures<>
{
  if constexpr (sizeof...(_Values) == 1)
  {
    throw __sender_type_check_failure<_Values..., _What...>(__values...);
  }
  else
  {
    throw __sender_type_check_failure<::cuda::std::__tuple<_Values...>, _What...>(::cuda::std::__tuple{__values...});
  }
}

#else // ^^^ constexpr exceptions ^^^ / vvv no constexpr exceptions vvv

template <class... _What, class... _Values>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto invalid_completion_signature(_Values...)
{
  return _ERROR<_What...>{};
}

#endif // ^^^ no constexpr exceptions ^^^

} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // _CUDAX_EXECUTION_COMPLETION_SIGNATURES_H
