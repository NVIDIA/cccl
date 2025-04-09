//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_COMPLETION_SIGNATURES
#define __CUDAX_ASYNC_DETAIL_COMPLETION_SIGNATURES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/typeid.h>
#include <cuda/std/tuple>

#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/exception.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/sender/type_traits.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <nv/target>

// include this last:
#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
using _CUDA_VSTD::__type_list;

enum class __stop_kind
{
  __unstoppable,
  __stoppable
};

template <class _ValueTuplesList = _CUDA_VSTD::__type_list<>,
          class _ErrorsList      = _CUDA_VSTD::__type_list<>,
          __stop_kind _StopKind  = __stop_kind::__unstoppable>
struct __partitioned_completions;

template <class _Tag>
struct __partitioned_fold_fn;

// The following overload of binary operator* is used to build up the cache of
// completion signatures. We fold over operator*, accumulating the completion
// signatures in the cache. `__undefined` is used here to prevent the
// instantiation of the intermediate types.
template <class _Partitioned, class _Tag, class... _Args>
_CUDAX_API auto operator*(__undefined<_Partitioned>&, _Tag (*)(_Args...))
  -> _CUDA_VSTD::__call_result_t<__partitioned_fold_fn<_Tag>, _Partitioned&, __undefined<__type_list<_Args...>>&>;

// This unary overload is used to extract the cache from the `__undefined` type.
template <class _Partitioned>
_CUDAX_API auto __unpack_partitioned_completions(__undefined<_Partitioned>&) -> _Partitioned;

template <class... _Sigs>
using __partition_completion_signatures_t _CCCL_NODEBUG_ALIAS = //
  decltype(__async::__unpack_partitioned_completions(
    (declval<__undefined<__partitioned_completions<>>&>() * ... * static_cast<_Sigs*>(nullptr))));

struct __concat_completion_signatures_helper;

template <class... _Sigs>
using __concat_completion_signatures_t =
  _CUDA_VSTD::__call_result_t<_CUDA_VSTD::__call_result_t<__concat_completion_signatures_helper, const _Sigs&...>>;

struct __concat_completion_signatures_fn
{
  template <class... _Sigs>
  _CUDAX_TRIVIAL_API constexpr auto operator()(const _Sigs&...) const noexcept
    -> __concat_completion_signatures_t<_Sigs...>
  {
    return {};
  }
};

_CCCL_GLOBAL_CONSTANT __concat_completion_signatures_fn concat_completion_signatures{};

#if defined(__cpp_constexpr_exceptions) // C++26, https://wg21.link/p3068
template <class... What, class... Values>
[[noreturn, nodiscard]] constexpr completion_signatures<> invalid_completion_signature(Values... values);
#else
template <class... What, class... Values>
[[nodiscard]] _CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto invalid_completion_signature(Values... values);
#endif

struct _IN_COMPLETION_SIGNATURES_APPLY;
struct _IN_COMPLETION_SIGNATURES_TRANSFORM_REDUCE;
struct _FUNCTION_IS_NOT_CALLABLE_WITH_THESE_SIGNATURES;

template <class _Fn, class _Sig>
using __completion_if =
  _CUDA_VSTD::_If<_CUDA_VSTD::__is_callable_v<_Fn, _Sig*>, completion_signatures<_Sig>, completion_signatures<>>;

template <class _Fn, class _Sig>
_CUDAX_API constexpr auto __filer_one(_Fn __fn, _Sig* __sig) -> __completion_if<_Fn, _Sig>
{
  if constexpr (_CUDA_VSTD::__is_callable_v<_Fn, _Sig*>)
  {
    __fn(__sig);
  }
  return {};
}

// working around compiler bugs in gcc and msvc
template <class... _Sigs>
using __completion_signatures _CCCL_NODEBUG_ALIAS = completion_signatures<_Sigs...>;

// A typelist for completion signatures
template <class... _Sigs>
struct _CCCL_TYPE_VISIBILITY_DEFAULT completion_signatures
{
  template <template <class...> class _Fn, template <class...> class _Continuation = __completion_signatures>
  using __transform_q _CCCL_NODEBUG_ALIAS = _Continuation<_CUDA_VSTD::__type_apply_q<_Fn, _Sigs>...>;

  template <class _Fn, class _Continuation = _CUDA_VSTD::__type_quote<__completion_signatures>>
  using __transform _CCCL_NODEBUG_ALIAS = __transform_q<_Fn::template __call, _Continuation::template __call>;

  using __partitioned _CCCL_NODEBUG_ALIAS = __partition_completion_signatures_t<_Sigs...>;

  template <class _Fn, class... _More>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call<_Fn, _Sigs..., _More...>;

  _CUDAX_DEFAULTED_API constexpr completion_signatures() = default;

  template <class _Tag>
  [[nodiscard]]
  _CUDAX_API constexpr auto count(_Tag) const noexcept -> size_t
  {
    if constexpr (_Tag() == set_value)
    {
      return __partitioned::__count_values::value;
    }
    else if constexpr (_Tag() == set_error)
    {
      return __partitioned::__count_errors::value;
    }
    else
    {
      return __partitioned::__count_stopped::value;
    }
  }

  template <class _Fn>
  _CUDAX_API constexpr auto apply(_Fn __fn) const -> _CUDA_VSTD::__call_result_t<_Fn, _Sigs*...>
  {
    return __fn(static_cast<_Sigs*>(nullptr)...);
  }

  template <class _Fn>
  [[nodiscard]]
  _CUDAX_API constexpr auto filter(_Fn __fn) const -> __concat_completion_signatures_t<__completion_if<_Fn, _Sigs>...>
  {
    return concat_completion_signatures(__async::__filer_one(__fn, static_cast<_Sigs*>(nullptr))...);
  }

  template <class _Tag>
  [[nodiscard]]
  _CUDAX_API constexpr auto select(_Tag) const noexcept
  {
    if constexpr (_Tag() == set_value)
    {
      return __partitioned::template __value_types<__type_function<set_value_t>::__call, completion_signatures>();
    }
    else if constexpr (_Tag() == set_error)
    {
      return __partitioned::template __error_types<completion_signatures, __type_function<set_error_t>::__call>();
    }
    else
    {
      return __partitioned::template __stopped_types<completion_signatures>();
    }
  }

  template <class _Transform, class _Reduce>
  [[nodiscard]]
  _CUDAX_API constexpr auto transform_reduce(_Transform __transform, _Reduce __reduce) const
    -> _CUDA_VSTD::__call_result_t<_Reduce, _CUDA_VSTD::__call_result_t<_Transform, _Sigs*>...>
  {
    return __reduce(__transform(static_cast<_Sigs*>(nullptr))...);
  }

  template <class... _OtherSigs>
  [[nodiscard]]
  _CUDAX_API constexpr auto operator+(completion_signatures<_OtherSigs...> __other) const noexcept
  {
    if constexpr (sizeof...(_OtherSigs) == 0) // short-circuit some common cases
    {
      (void) __other;
      return *this;
    }
    else if constexpr (sizeof...(_Sigs) == 0)
    {
      return __other;
    }
    else
    {
      return concat_completion_signatures(*this, __other);
    }
  }
};

completion_signatures() -> completion_signatures<>;

template <class _Ty>
_CCCL_CONCEPT __valid_completion_signatures = detail::__is_specialization_of<_Ty, completion_signatures>;

template <class _Derived>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __compile_time_error // : ::std::exception
{
  _CUDAX_DEFAULTED_API __compile_time_error() = default;

  const char* what() const noexcept // override
  {
    return _CCCL_TYPEID(_Derived*).name();
  }
};

template <class _Data, class... _What>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sender_type_check_failure
    : __compile_time_error<__sender_type_check_failure<_Data, _What...>>
{
  _CUDAX_DEFAULTED_API __sender_type_check_failure() = default;

  explicit constexpr __sender_type_check_failure(_Data data)
      : data_(data)
  {}

  _Data data_{};
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT dependent_sender_error // : ::std::exception
{
  _CUDAX_TRIVIAL_API char const* what() const noexcept // override
  {
    return what_;
  }

  char const* what_;
};

template <class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __dependent_sender_error : dependent_sender_error
{
  _CUDAX_TRIVIAL_API constexpr __dependent_sender_error() noexcept
      : dependent_sender_error{"This sender needs to know its execution " //
                               "environment before it can know how it will complete."}
  {}

  _CCCL_HOST_DEVICE __dependent_sender_error operator+();

  template <class _Ty>
  _CCCL_HOST_DEVICE __dependent_sender_error& operator,(_Ty&);

  template <class... _What>
  _CCCL_HOST_DEVICE _ERROR<_What...>&operator,(_ERROR<_What...>&);
};

// Below is the definition of the _CUDAX_LET_COMPLETIONS portability macro. It
// is used to check that an expression's type is a valid completion_signature
// specialization.
//
// USAGE:
//
//   _CUDAX_LET_COMPLETIONS(auto(__cs) = <expression>)
//   {
//     // __cs is guaranteed to be a specialization of completion_signatures.
//   }
//
// When constexpr exceptions are available (C++26), the macro simply expands to
// the moral equivalent of:
//
//   // With constexpr exceptions:
//   auto __cs = <expression>; // throws if __cs is not a completion_signatures
//
// When constexpr exceptions are not available, the macro expands to:
//
//   // Without constexpr exceptions:
//   if constexpr (auto __cs = <expression>; !__valid_completion_signatures<decltype(__cs)>)
//   {
//     return __cs;
//   }
//   else

#if defined(__cpp_constexpr_exceptions) // C++26, https://wg21.link/p3068

#  define _CUDAX_LET_COMPLETIONS(...)                  \
    if constexpr ([[maybe_unused]] __VA_ARGS__; false) \
    {                                                  \
    }                                                  \
    else

template <class... What, class... Values>
[[noreturn, nodiscard]] _CUDAX_API consteval completion_signatures<> invalid_completion_signature(Values... values)
{
  if constexpr (sizeof...(Values) == 1)
  {
    throw __sender_type_check_failure<Values..., What...>(values...);
  }
  else
  {
    throw __sender_type_check_failure<__tuple<Values...>, What...>(__tupl{values...});
  }
}

template <class... _Sndr>
[[noreturn, nodiscard]] _CUDAX_API consteval auto __dependent_sender() -> completion_signatures<>
{
  throw __dependent_sender_error<_Sndr...>();
}

#else // ^^^ constexpr exceptions ^^^ / vvv no constexpr exceptions vvv

#  define _CUDAX_PP_EAT_AUTO_auto(_ID)    _ID _CCCL_PP_EAT _CCCL_PP_LPAREN
#  define _CUDAX_PP_EXPAND_AUTO_auto(_ID) auto _ID
#  define _CUDAX_LET_COMPLETIONS_ID(...)  _CCCL_PP_EXPAND(_CCCL_PP_CAT(_CUDAX_PP_EAT_AUTO_, __VA_ARGS__) _CCCL_PP_RPAREN)

#  define _CUDAX_LET_COMPLETIONS(...)                                                                               \
    if constexpr (_CCCL_PP_CAT(_CUDAX_PP_EXPAND_AUTO_, __VA_ARGS__);                                                \
                  !::cuda::experimental::__async::__valid_completion_signatures<decltype(_CUDAX_LET_COMPLETIONS_ID( \
                    __VA_ARGS__))>)                                                                                 \
    {                                                                                                               \
      return _CUDAX_LET_COMPLETIONS_ID(__VA_ARGS__);                                                                \
    }                                                                                                               \
    else

template <class... What, class... Values>
[[nodiscard]] _CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto invalid_completion_signature([[maybe_unused]] Values... values)
{
  return _ERROR<What...>();
}

template <class... _Sndr>
[[nodiscard]] _CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto __dependent_sender() -> __dependent_sender_error<_Sndr...>
{
  return __dependent_sender_error<_Sndr...>();
}

#endif // ^^^ no constexpr exceptions ^^^

_CCCL_DIAG_PUSH
// warning C4913: user defined binary operator ',' exists but no overload could convert all operands,
// default built-in binary operator ',' used
_CCCL_DIAG_SUPPRESS_MSVC(4913)

#define _CUDAX_GET_COMPLSIGS(...) \
  _CUDA_VSTD::remove_reference_t<_Sndr>::template get_completion_signatures<__VA_ARGS__>()

#define _CUDAX_CHECKED_COMPLSIGS(...) (__VA_ARGS__, void(), __async::__checked_complsigs<decltype(__VA_ARGS__)>())

struct _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION
{};

template <class _Completions>
_CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto __checked_complsigs()
{
  _CUDAX_LET_COMPLETIONS(auto(__cs) = _Completions())
  {
    if constexpr (__valid_completion_signatures<_Completions>)
    {
      return __cs;
    }
    else
    {
      return invalid_completion_signature<
        _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
        _WITH_SIGNATURES(_Completions)>();
    }
  }
}

template <class _Sndr, class... _Env>
inline constexpr bool __has_get_completion_signatures = false;

// clang-format off
template <class _Sndr>
inline constexpr bool __has_get_completion_signatures<_Sndr> =
  _CCCL_REQUIRES_EXPR((_Sndr))
  (
    (_CUDAX_GET_COMPLSIGS(_Sndr))
  );

template <class _Sndr, class _Env>
inline constexpr bool __has_get_completion_signatures<_Sndr, _Env> =
  _CCCL_REQUIRES_EXPR((_Sndr, _Env))
  (
    (_CUDAX_GET_COMPLSIGS(_Sndr, _Env))
  );
// clang-format on

struct _COULD_NOT_DETERMINE_COMPLETION_SIGNATURES_FOR_THIS_SENDER
{};

template <class _Sndr, class... _Env>
_CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto __get_completion_signatures_helper()
{
  if constexpr (__has_get_completion_signatures<_Sndr, _Env...>)
  {
    return _CUDAX_CHECKED_COMPLSIGS(_CUDAX_GET_COMPLSIGS(_Sndr, _Env...));
  }
  else if constexpr (__has_get_completion_signatures<_Sndr>)
  {
    return _CUDAX_CHECKED_COMPLSIGS(_CUDAX_GET_COMPLSIGS(_Sndr));
  }
  // else if constexpr (__is_awaitable<_Sndr, __env_promise<_Env>...>)
  // {
  //   using Result = __await_result_t<_Sndr, __env_promise<_Env>...>;
  //   return completion_signatures{__set_value_v<Result>, __set_error_v<>, __set_stopped_v};
  // }
  else if constexpr (sizeof...(_Env) == 0)
  {
    return __dependent_sender<_Sndr>();
  }
  else
  {
    return invalid_completion_signature<_COULD_NOT_DETERMINE_COMPLETION_SIGNATURES_FOR_THIS_SENDER,
                                        _WITH_SENDER(_Sndr),
                                        _WITH_ENVIRONMENT(_Env...)>();
  }
}

template <class _Sndr, class... _Env>
_CUDAX_TRIVIAL_API _CUDAX_CONSTEVAL auto get_completion_signatures()
{
  static_assert(sizeof...(_Env) <= 1, "At most one environment is allowed.");
  if constexpr (0 == sizeof...(_Env))
  {
    return __get_completion_signatures_helper<_Sndr>();
  }
  else
  {
    // BUGBUG TODO:
    // Apply a lazy sender transform if one exists before computing the completion signatures:
    // using _NewSndr = __transform_sender_result_t<__late_domain_of_t<_Sndr, _Env>, _Sndr, _Env>;
    using _NewSndr = _Sndr;
    return __get_completion_signatures_helper<_NewSndr, _Env...>();
  }
}

// BUGBUG TODO
template <class _Env>
using _FWD_ENV_T = _Env;

template <class _Parent, class _Child, class... _Env>
constexpr auto get_child_completion_signatures()
{
  return get_completion_signatures<__copy_cvref_t<_Parent, _Child>, _FWD_ENV_T<_Env>...>();
}

#undef _CUDAX_GET_COMPLSIGS
#undef _CUDAX_CHECKED_COMPLSIGS
_CCCL_DIAG_POP

template <class _Completions>
using __partitioned_completions_of = typename _Completions::__partitioned;

constexpr int __invalid_disposition = -1;

// A metafunction to determine whether a type is a completion signature, and if
// so, what its disposition is.
template <class>
inline constexpr int __signature_disposition = __invalid_disposition;

template <class... _Values>
inline constexpr __disposition_t __signature_disposition<set_value_t(_Values...)> = __disposition_t::__value;

template <class _Error>
inline constexpr __disposition_t __signature_disposition<set_error_t(_Error)> = __disposition_t::__error;

template <>
inline constexpr __disposition_t __signature_disposition<set_stopped_t()> = __disposition_t::__stopped;

template <class _WantedTag>
struct __gather_sigs_fn;

template <>
struct __gather_sigs_fn<set_value_t>
{
  template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
  using __call = typename __partitioned_completions_of<_Sigs>::template __value_types<_Tuple, _Variant>;
};

template <>
struct __gather_sigs_fn<set_error_t>
{
  template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
  using __call = typename __partitioned_completions_of<_Sigs>::template __error_types<_Variant, _Tuple>;
};

template <>
struct __gather_sigs_fn<set_stopped_t>
{
  template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
  using __call = typename __partitioned_completions_of<_Sigs>::template __stopped_types<_Variant, _Tuple<>>;
};

template <class _Sigs, class _WantedTag, template <class...> class _TaggedTuple, template <class...> class _Variant>
using __gather_completion_signatures =
  typename __gather_sigs_fn<_WantedTag>::template __call<_Sigs, _TaggedTuple, _Variant>;

// __partitioned_completions is a cache of completion signatures for fast
// access. The completion_signatures<Sigs...>::__partitioned nested struct
// inherits from __partitioned_completions. If the cache is never accessed,
// it is never instantiated.
template <class... _ValueTuples, class... _Errors, __stop_kind _StopKind>
struct __partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>, _CUDA_VSTD::__type_list<_Errors...>, _StopKind>
{
  template <template <class...> class _Tuple, template <class...> class _Variant>
  using __value_types = _Variant<_CUDA_VSTD::__type_call1<_ValueTuples, _CUDA_VSTD::__type_quote<_Tuple>>...>;

  template <template <class...> class _Variant, template <class> class _Transform = _CUDA_VSTD::__type_self_t>
  using __error_types = _Variant<_Transform<_Errors>...>;

  template <template <class...> class _Variant, class _Type = set_stopped_t()>
  using __stopped_types = _CUDA_VSTD::__type_call1<
    _CUDA_VSTD::_If<_StopKind == __stop_kind::__stoppable, _CUDA_VSTD::__type_list<_Type>, _CUDA_VSTD::__type_list<>>,
    _CUDA_VSTD::__type_quote<_Variant>>;

  using __count_values  = _CUDA_VSTD::integral_constant<size_t, sizeof...(_ValueTuples)>;
  using __count_errors  = _CUDA_VSTD::integral_constant<size_t, sizeof...(_Errors)>;
  using __count_stopped = _CUDA_VSTD::integral_constant<size_t, _StopKind == __stop_kind::__stoppable ? 1 : 0>;

  struct __nothrow_decay_copyable
  {
    // These aliases are placed in a separate struct to avoid computing them
    // if they are not needed.
    using __fn     = _CUDA_VSTD::__type_quote<__nothrow_decay_copyable_t>;
    using __values = _CUDA_VSTD::_And<_CUDA_VSTD::__type_call1<_ValueTuples, __fn>...>;
    using __errors = __nothrow_decay_copyable_t<_Errors...>;
    using __all    = _CUDA_VSTD::_And<__values, __errors>;
  };
};

template <>
struct __partitioned_fold_fn<set_value_t>
{
  template <class... _ValueTuples, class _Errors, __stop_kind _StopKind, class _Values>
  _CUDAX_API auto operator()(__partitioned_completions<__type_list<_ValueTuples...>, _Errors, _StopKind>&,
                             __undefined<_Values>&) const
    -> __undefined<__partitioned_completions<__type_list<_ValueTuples..., _Values>, _Errors, _StopKind>>&;
};

template <>
struct __partitioned_fold_fn<set_error_t>
{
  template <class _Values, class... _Errors, __stop_kind _StopKind, class _Error>
  _CUDAX_API auto operator()(__partitioned_completions<_Values, __type_list<_Errors...>, _StopKind>&,
                             __undefined<__type_list<_Error>>&) const
    -> __undefined<__partitioned_completions<_Values, __type_list<_Errors..., _Error>, _StopKind>>&;
};

template <>
struct __partitioned_fold_fn<set_stopped_t>
{
  template <class _Values, class _Errors, __stop_kind _StopKind>
  _CUDAX_API auto operator()(__partitioned_completions<_Values, _Errors, _StopKind>&, detail::__ignore) const
    -> __undefined<__partitioned_completions<_Values, _Errors, __stop_kind::__stoppable>>&;
};

// make_completion_signatures
template <class _Tag, class... _As>
_CUDAX_API auto __normalize_helper(_As&&...) -> _Tag (*)(_As...);

template <class _Tag, class... _As>
_CUDAX_API auto __normalize(_Tag (*)(_As...)) -> decltype(__async::__normalize_helper<_Tag>(declval<_As>()...));

template <class... _Sigs>
_CUDAX_API auto __make_unique(_Sigs*...)
  -> _CUDA_VSTD::__type_apply<_CUDA_VSTD::__type_quote<completion_signatures>, _CUDA_VSTD::__make_type_set<_Sigs...>>;

template <class... _Sigs>
using __make_completion_signatures_t =
  decltype(__async::__make_unique(__async::__normalize(static_cast<_Sigs*>(nullptr))...));

template <class... _ExplicitSigs, class... _DeducedSigs>
_CUDAX_TRIVIAL_API constexpr auto make_completion_signatures(_DeducedSigs*...) noexcept
  -> __make_completion_signatures_t<_ExplicitSigs..., _DeducedSigs...>
{
  return {};
}

// concat_completion_signatures
extern const completion_signatures<>& __empty_completion_signatures;

struct __concat_completion_signatures_helper
{
  _CUDAX_TRIVIAL_API constexpr auto operator()() const noexcept -> completion_signatures<> (*)()
  {
    return nullptr;
  }

  template <class... _Sigs>
  _CUDAX_TRIVIAL_API constexpr auto operator()(const completion_signatures<_Sigs...>&) const noexcept
    -> __make_completion_signatures_t<_Sigs...> (*)()
  {
    return nullptr;
  }

  template <class _Self = __concat_completion_signatures_helper,
            class... _As,
            class... _Bs,
            class... _Cs,
            class... _Ds,
            class... _Rest>
  _CUDAX_TRIVIAL_API constexpr auto operator()(
    const completion_signatures<_As...>&,
    const completion_signatures<_Bs...>&,
    const completion_signatures<_Cs...>& = __empty_completion_signatures,
    const completion_signatures<_Ds...>& = __empty_completion_signatures,
    const _Rest&...) const noexcept
  {
    using _Tmp       = completion_signatures<_As..., _Bs..., _Cs..., _Ds...>;
    using _SigsFnPtr = _CUDA_VSTD::__call_result_t<_Self, const _Tmp&, const _Rest&...>;
    return static_cast<_SigsFnPtr>(nullptr);
  }

  template <class _Ap, class _Bp = __ignore, class _Cp = __ignore, class _Dp = __ignore, class... _Rest>
  _CUDAX_TRIVIAL_API constexpr auto
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

template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
using __value_types = typename _Sigs::__partitioned::template __value_types<_Tuple, _Variant>;

template <class _Sndr, class _Env, template <class...> class _Tuple, template <class...> class _Variant>
using value_types_of_t =
  __value_types<completion_signatures_of_t<_Sndr, _Env>, _Tuple, __type_try_quote<_Variant>::template __call>;

template <class _Sigs, template <class...> class _Variant>
using __error_types = typename _Sigs::__partitioned::template __error_types<_Variant, _CUDA_VSTD::__type_self_t>;

template <class _Sndr, class _Env, template <class...> class _Variant>
using error_types_of_t = __error_types<completion_signatures_of_t<_Sndr, _Env>, _Variant>;

template <class _Sigs, template <class...> class _Variant, class _Type>
using __stopped_types = typename _Sigs::__partitioned::template __stopped_types<_Variant, _Type>;

template <class _Sigs>
inline constexpr bool __sends_stopped = _Sigs::__partitioned::__count_stopped::value != 0;

template <class _Sndr, class... _Env>
inline constexpr bool sends_stopped = __sends_stopped<completion_signatures_of_t<_Sndr, _Env...>>;

template <class _Tag>
struct __default_transform_fn
{
  template <class... _Ts>
  _CUDAX_TRIVIAL_API constexpr auto operator()() const noexcept -> completion_signatures<_Tag(_Ts...)>
  {
    return {};
  }
};

struct __swallow_transform
{
  template <class... _Ts>
  _CUDAX_TRIVIAL_API constexpr auto operator()() const noexcept -> completion_signatures<>
  {
    return {};
  }
};

template <class _Tag>
struct __decay_transform
{
  template <class... _Ts>
  _CUDAX_TRIVIAL_API constexpr auto operator()() const noexcept
    -> completion_signatures<_Tag(_CUDA_VSTD::decay_t<_Ts>...)>
  {
    return {};
  }
};

template <class _Fn, class... _As>
using __meta_call_result_t = decltype(declval<_Fn>().template operator()<_As...>());

template <class _Ay, class... _As, class _Fn>
_CUDAX_TRIVIAL_API constexpr auto __transform_expr(const _Fn& __fn) -> __meta_call_result_t<const _Fn&, _Ay, _As...>
{
  return __fn.template operator()<_Ay, _As...>();
}

template <class _Fn>
_CUDAX_TRIVIAL_API constexpr auto __transform_expr(const _Fn& __fn) -> __call_result_t<const _Fn&>
{
  return __fn();
}

template <class _Fn, class... _As>
using __transform_expr_t = decltype(__async::__transform_expr<_As...>(declval<const _Fn&>()));

struct _IN_TRANSFORM_COMPLETION_SIGNATURES;
struct _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION;
struct _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS;

// transform_completion_signatures:
template <class... _As, class _Fn>
_CUDAX_API constexpr auto __apply_transform(const _Fn& __fn)
{
  if constexpr (__type_valid_v<__transform_expr_t, _Fn, _As...>)
  {
    using __completions = __transform_expr_t<_Fn, _As...>;
    if constexpr (__valid_completion_signatures<__completions> || __type_is_error<__completions>
                  || _CUDA_VSTD::is_base_of_v<dependent_sender_error, __completions>)
    {
      return __async::__transform_expr<_As...>(__fn);
    }
    else
    {
      (void) __async::__transform_expr<_As...>(__fn); // potentially throwing
      return invalid_completion_signature<
        _IN_TRANSFORM_COMPLETION_SIGNATURES,
        _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
        _WITH_FUNCTION(_Fn),
        _WITH_ARGUMENTS(_As...)>();
    }
  }
  else
  {
    return invalid_completion_signature< //
      _IN_TRANSFORM_COMPLETION_SIGNATURES,
      _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS,
      _WITH_FUNCTION(_Fn),
      _WITH_ARGUMENTS(_As...)>();
  }
}

template <class _ValueFn, class _ErrorFn, class _StoppedFn>
struct __transform_one
{
  _ValueFn __value_fn;
  _ErrorFn __error_fn;
  _StoppedFn __stopped_fn;

  template <class _Tag, class... _Ts>
  _CUDAX_API constexpr auto operator()(_Tag (*)(_Ts...)) const
  {
    if constexpr (_Tag() == set_value)
    {
      return __apply_transform<_Ts...>(__value_fn);
    }
    else if constexpr (_Tag() == set_error)
    {
      return __apply_transform<_Ts...>(__error_fn);
    }
    else
    {
      return __apply_transform<_Ts...>(__stopped_fn);
    }
  }
};

template <class _TransformOne>
struct __transform_all_fn
{
  _TransformOne __tfx1;

  template <class... _Sigs>
  _CUDAX_API constexpr auto operator()(_Sigs*... __sigs) const
  {
    return concat_completion_signatures(__tfx1(__sigs)...);
  }
};

template <class _TransformOne>
__transform_all_fn(_TransformOne) -> __transform_all_fn<_TransformOne>;

template <class _Completions,
          class _ValueFn   = __default_transform_fn<set_value_t>,
          class _ErrorFn   = __default_transform_fn<set_error_t>,
          class _StoppedFn = __default_transform_fn<set_stopped_t>,
          class _ExtraSigs = completion_signatures<>>
_CUDAX_API constexpr auto transform_completion_signatures(
  _Completions, //
  _ValueFn __value_fn     = {},
  _ErrorFn __error_fn     = {},
  _StoppedFn __stopped_fn = {},
  _ExtraSigs              = {})
{
  _CUDAX_LET_COMPLETIONS(auto(__completions) = _Completions())
  {
    _CUDAX_LET_COMPLETIONS(auto(__extra) = _ExtraSigs())
    {
      __transform_one<_ValueFn, _ErrorFn, _StoppedFn> __tfx1{__value_fn, __error_fn, __stopped_fn};
      return concat_completion_signatures(__completions.apply(__transform_all_fn{__tfx1}), __extra);
    }
  }
}

#if _CCCL_HAS_EXCEPTIONS()
_CUDAX_API inline constexpr auto __eptr_completion() noexcept
{
  return completion_signatures<set_error_t(::std::exception_ptr)>();
}
#else  // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
_CUDAX_API inline constexpr auto __eptr_completion() noexcept
{
  return completion_signatures{};
}
#endif // !_CCCL_HAS_EXCEPTIONS()

template <bool _PotentiallyThrowing>
_CUDAX_API constexpr auto __eptr_completion_if() noexcept
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

#if defined(__cpp_constexpr_exceptions) // C++26, https://wg21.link/p3068
// When asked for its completions without an envitonment, a dependent sender
// will throw an exception of a type derived from `dependent_sender_error`.
template <class _Sndr>
_CUDAX_API constexpr bool __is_dependent_sender() noexcept
try
{
  (void) get_completion_signatures<_Sndr>();
  return false; // didn't throw, not a dependent sender
}
catch (dependent_sender_error&)
{
  return true;
}
catch (...)
{
  return false; // different kind of exception was thrown; not a dependent sender
}
#else
template <class _Sndr>
_CUDAX_API constexpr bool __is_dependent_sender() noexcept
{
  using _Completions = decltype(get_completion_signatures<_Sndr>());
  return _CUDA_VSTD::is_base_of_v<dependent_sender_error, _Completions>;
}
#endif

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
