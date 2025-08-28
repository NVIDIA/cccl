//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_META
#define __CUDAX_EXECUTION_META

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/utility.cuh>

#if __cpp_lib_three_way_comparison
#  include <compare> // IWYU pragma: keep
#endif

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-string-literal-operator-template")
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")

namespace cuda::experimental::execution
{
// The following must be left undefined
template <class...>
struct _DIAGNOSTIC;

struct _UNKNOWN;

struct _WHERE;

struct _WHAT;

struct _IN_ALGORITHM;

struct _WITH_FUNCTION;

struct _WITH_SENDER;

struct _WITH_ARGUMENTS;

struct _WITH_QUERY;

struct _WITH_ENVIRONMENT;

struct _WITH_SIGNATURES;

template <class>
struct _WITH_COMPLETION_SIGNATURE;

struct _FUNCTION_IS_NOT_CALLABLE;

struct _FUNCTION_MUST_RETURN_A_SENDER;

struct _FUNCTION_MUST_RETURN_SENDERS_THAT_ALL_COMPLETE_IN_A_COMMON_DOMAIN;

struct _WITH_RETURN_TYPE;

struct _SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS;

struct _ARGUMENTS_ARE_NOT_DECAY_COPYABLE;

struct _THE_ENVIRONMENT_OF_THE_RECEIVER_DOES_NOT_HAVE_A_SCHEDULER_FOR_ON_TO_RETURN_TO;

struct __merror_base
{
  // _CCCL_HIDE_FROM_ABI virtual ~__merror_base() = default;

  _CCCL_HOST_DEVICE friend constexpr auto __ustdex_unhandled_error(void*) noexcept -> bool
  {
    return true;
  }
};

template <class... _What>
struct _ERROR : __merror_base
{
  // The following aliases are to simplify error propagation
  // in the completion signatures meta-programming.
  template <class...>
  using __call _CCCL_NODEBUG_ALIAS = _ERROR;

  using __partitioned _CCCL_NODEBUG_ALIAS = _ERROR;

  template <template <class...> class, template <class...> class>
  using __value_types _CCCL_NODEBUG_ALIAS = _ERROR;

  template <template <class...> class>
  using __error_types _CCCL_NODEBUG_ALIAS = _ERROR;

  using __sends_stopped _CCCL_NODEBUG_ALIAS = _ERROR;

  // The following operator overloads also simplify error propagation.
  _CCCL_HOST_DEVICE auto operator+() -> _ERROR;

  template <class _Ty>
  _CCCL_HOST_DEVICE auto operator,(_Ty&) -> _ERROR&;

  template <class... _With>
  _CCCL_HOST_DEVICE auto with(_ERROR<_With...>&) -> _ERROR<_What..., _With...>&;
};

_CCCL_HOST_DEVICE constexpr auto __ustdex_unhandled_error(...) noexcept -> bool
{
  return false;
}

template <class _Ty>
inline constexpr bool __type_is_error = false;

template <class... _What>
inline constexpr bool __type_is_error<_ERROR<_What...>> = true;

template <class... _What>
inline constexpr bool __type_is_error<_ERROR<_What...>&> = true;

// True if any of the types in _Ts... are errors; false otherwise.
template <class... _Ts>
inline constexpr bool __type_contains_error =
#if _CCCL_COMPILER(MSVC)
  (__type_is_error<_Ts> || ...);
#else
  __ustdex_unhandled_error(static_cast<::cuda::std::__type_list<_Ts...>*>(nullptr));
#endif

template <class... _Ts>
using __type_find_error _CCCL_NODEBUG_ALIAS = decltype(+(declval<_Ts&>(), ..., declval<_ERROR<_UNKNOWN>&>()));

template <bool _Error>
struct __type_self_or_error_with_
{
  template <class _Ty, class... _With>
  using __call _CCCL_NODEBUG_ALIAS = _Ty;
};

template <>
struct __type_self_or_error_with_<true>
{
  template <class _Ty, class... _With>
  using __call _CCCL_NODEBUG_ALIAS = decltype(declval<_Ty&>().with(declval<_ERROR<_With...>&>()));
};

template <class _Ty, class... _With>
using __type_self_or_error_with _CCCL_NODEBUG_ALIAS =
  ::cuda::std::__type_call<__type_self_or_error_with_<__type_is_error<_Ty>>, _Ty, _With...>;

template <bool>
struct __type_try__;

template <>
struct __type_try__<false>
{
  template <template <class...> class _Fn, class... _Ts>
  using __call_q _CCCL_NODEBUG_ALIAS = _Fn<_Ts...>;

  template <class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = typename _Fn::template __call<_Ts...>;
};

template <>
struct __type_try__<true>
{
  template <template <class...> class _Fn, class... _Ts>
  using __call_q _CCCL_NODEBUG_ALIAS = __type_find_error<_Ts...>;

  template <class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_find_error<_Fn, _Ts...>;
};

template <class _Fn, class... _Ts>
using __type_try_call _CCCL_NODEBUG_ALIAS =
  typename __type_try__<__type_contains_error<_Fn, _Ts...>>::template __call<_Fn, _Ts...>;

template <template <class...> class _Fn, class... _Ts>
using __type_try_call_quote _CCCL_NODEBUG_ALIAS =
  typename __type_try__<__type_contains_error<_Ts...>>::template __call_q<_Fn, _Ts...>;

// wraps a meta-callable such that if any of the arguments are errors, the
// result is an error.
template <class _Fn>
struct __type_try
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_try_call<_Fn, _Ts...>;
};

template <template <class...> class _Fn, class... _Default>
struct __type_try_quote;

// equivalent to __type_try<__type_quote<_Fn>>
template <template <class...> class _Fn>
struct __type_try_quote<_Fn>
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS =
    typename __type_try__<__type_contains_error<_Ts...>>::template __call_q<_Fn, _Ts...>;
};

// equivalent to __type_try<__type_quote<_Fn, _Default>>
template <template <class...> class _Fn, class _Default>
struct __type_try_quote<_Fn, _Default>
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS =
    typename ::cuda::std::_If<__is_instantiable_with<_Fn, _Ts...>, //
                              __type_try_quote<_Fn>,
                              ::cuda::std::__type_always<_Default>>::template __call<_Ts...>;
};

template <class _Return>
struct __type_function
{
  template <class... _Args>
  using __call _CCCL_NODEBUG_ALIAS = _Return(_Args...);
};

template <class _Return>
struct __type_function1
{
  template <class _Arg>
  using __call _CCCL_NODEBUG_ALIAS = _Return(_Arg);
};

template <class _First, class _Second>
using __type_first = _First;

template <class _First, class _Second>
using __type_second = _Second;

template <template <class...> class _Second, template <class...> class _First>
struct __type_compose_quote
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = _Second<_First<_Ts...>>;
};

struct __type_count
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = ::cuda::std::integral_constant<size_t, sizeof...(_Ts)>;
};

template <class _Continuation>
struct __type_concat_into
{
  template <class... _Args>
  using __call _CCCL_NODEBUG_ALIAS =
    ::cuda::std::__type_call1<::cuda::std::__type_concat<::cuda::std::__as_type_list<_Args>...>, _Continuation>;
};

template <template <class...> class _Continuation>
struct __type_concat_into_quote : __type_concat_into<::cuda::std::__type_quote<_Continuation>>
{};

template <class _Ty>
struct __type_self_or
{
  template <class _Uy = _Ty>
  using __call _CCCL_NODEBUG_ALIAS = _Uy;
};

template <template <class...> class _Fn, class _Default, class... _Ts>
using __type_call_or_q =
  typename ::cuda::std::_If<__is_instantiable_with<_Fn, _Ts...>,
                            ::cuda::std::__type_quote<_Fn>,
                            ::cuda::std::__type_always<_Default>>::template __call<_Ts...>;

template <class _Fn, class _Default, class... _Ts>
using __type_call_or =
  typename ::cuda::std::_If<__is_instantiable_with<_Fn::template __call, _Ts...>,
                            _Fn,
                            ::cuda::std::__type_always<_Default>>::template __call<_Ts...>;

} // namespace cuda::experimental::execution

_CCCL_DIAG_POP

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_META
