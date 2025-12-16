//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_DIAGNOSTICS
#define __CUDAX_EXECUTION_DIAGNOSTICS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/fwd.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// The following must be left undefined
template <class...>
struct _DIAGNOSTIC;

struct _UNKNOWN;

struct _WHERE;

struct _WHAT;

struct _TO_FIX_THIS_ERROR;

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

template <class... _What>
struct __not_a_sender
{
  using sender_concept = sender_t;

  template <class...>
  _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return execution::invalid_completion_signature<_What...>();
  }
};

template <class... _What>
struct __not_a_scheduler
{
  using scheduler_concept = scheduler_t;

  _CCCL_API auto schedule() noexcept
  {
    return __not_a_sender<_What...>{};
  }

  _CCCL_API constexpr bool operator==(__not_a_scheduler) const noexcept
  {
    return true;
  }

  _CCCL_API constexpr bool operator!=(__not_a_scheduler) const noexcept
  {
    return false;
  }
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_DIAGNOSTICS
