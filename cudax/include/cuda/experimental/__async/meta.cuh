//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_META
#define __CUDAX_ASYNC_DETAIL_META

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

#include <cuda/experimental/__detail/config.cuh>

#if __cpp_lib_three_way_comparison
#  include <compare>
#endif

#include <cuda/experimental/__async/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-string-literal-operator-template")
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")

namespace cuda::experimental::__async
{
template <class _Ret, class... _Args>
using __fn_t = _Ret(_Args...);

template <class _Ty>
_Ty&& __declval() noexcept;

template <size_t... _Vals>
struct __moffsets;

// The following must be left undefined
template <class...>
struct _DIAGNOSTIC;

struct _WHERE;

struct _IN_ALGORITHM;

struct _WHAT;

struct _WITH_FUNCTION;

struct _WITH_SENDER;

struct _WITH_ARGUMENTS;

struct _WITH_QUERY;

struct _WITH_ENVIRONMENT;

template <class>
struct _WITH_COMPLETION_SIGNATURE;

struct _FUNCTION_IS_NOT_CALLABLE;

struct _UNKNOWN;

struct _SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS;

template <class... _Sigs>
struct _WITH_COMPLETIONS
{};

struct __merror_base
{
  constexpr friend bool __ustdex_unhandled_error(void*) noexcept
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

  using __partitions _CCCL_NODEBUG_ALIAS = _ERROR;

  template <template <class...> class, template <class...> class>
  using __value_types _CCCL_NODEBUG_ALIAS = _ERROR;

  template <template <class...> class>
  using __error_types _CCCL_NODEBUG_ALIAS = _ERROR;

  using __sends_stopped _CCCL_NODEBUG_ALIAS = _ERROR;

  // The following operator overloads also simplify error propagation.
  _ERROR operator+();

  template <class _Ty>
  _ERROR& operator,(_Ty&);

  template <class... _With>
  _ERROR<_What..., _With...>& with(_ERROR<_With...>&);
};

constexpr bool __ustdex_unhandled_error(...) noexcept
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
#if defined(_CCCL_COMPILER_MSVC)
  (__type_is_error<_Ts> || ...);
#else
  __ustdex_unhandled_error(static_cast<_CUDA_VSTD::__type_list<_Ts...>*>(nullptr));
#endif

template <class... _Ts>
using __type_find_error = decltype(+(__declval<_Ts&>(), ..., __declval<_ERROR<_UNKNOWN>&>()));

template <template <class...> class _Fn, class... _Ts>
inline constexpr bool __type_valid_v = _CUDA_VSTD::_IsValidExpansion<_Fn, _Ts...>::value;

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
  using __call _CCCL_NODEBUG_ALIAS = decltype(__declval<_Ty&>().with(__declval<_ERROR<_With...>&>()));
};

template <class _Ty, class... _With>
using __type_self_or_error_with =
  _CUDA_VSTD::__type_call<__type_self_or_error_with_<__type_is_error<_Ty>>, _Ty, _With...>;

template <bool>
struct __type_try__;

template <>
struct __type_try__<false>
{
  template <template <class...> class _Fn, class... _Ts>
  using __call_q = _Fn<_Ts...>;

  template <class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = typename _Fn::template __call<_Ts...>;
};

template <>
struct __type_try__<true>
{
  template <template <class...> class _Fn, class... _Ts>
  using __call_q = __type_find_error<_Ts...>;

  template <class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = __type_find_error<_Fn, _Ts...>;
};

template <class _Fn, class... _Ts>
using __type_try_call = typename __type_try__<__type_contains_error<_Fn, _Ts...>>::template __call<_Fn, _Ts...>;

template <template <class...> class _Fn, class... _Ts>
using __type_try_call_quote = typename __type_try__<__type_contains_error<_Ts...>>::template __call_q<_Fn, _Ts...>;

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
    typename _CUDA_VSTD::conditional_t<__type_valid_v<_Fn, _Ts...>, //
                                       __type_try_quote<_Fn>,
                                       _CUDA_VSTD::__type_always<_Default>>::template __call<_Ts...>;
};

template <class _First, class _Second>
struct __type_pair
{
  using first  = _First;
  using second = _Second;
};

template <class _Pair>
using __type_first = typename _Pair::first;

template <class _Pair>
using __type_second = typename _Pair::second;

template <template <class...> class _Second, template <class...> class _First>
struct __type_compose_quote
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = _Second<_First<_Ts...>>;
};

struct __type_count
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::integral_constant<size_t, sizeof...(_Ts)>;
};

template <class _Continuation>
struct __type_concat_into
{
  template <class... _Args>
  using __call _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::__type_call1<_CUDA_VSTD::__type_concat<_CUDA_VSTD::__as_type_list<_Args>...>, _Continuation>;
};

template <template <class...> class _Continuation>
struct __type_concat_into_quote : __type_concat_into<_CUDA_VSTD::__type_quote<_Continuation>>
{};

template <class _Ty>
struct __type_self_or
{
  template <class _Uy = _Ty>
  using __call _CCCL_NODEBUG_ALIAS = _Uy;
};

template <class _Ret>
struct __type_quote_function
{
  template <class... _Args>
  using __call _CCCL_NODEBUG_ALIAS = _Ret(_Args...);
};
} // namespace cuda::experimental::__async

_CCCL_DIAG_POP

#include <cuda/experimental/__async/epilogue.cuh>

#endif
