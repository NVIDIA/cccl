//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_READ_ENV
#define __CUDAX_EXECUTION_READ_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _THE_CURRENT_ENVIRONMENT_LACKS_THIS_QUERY;
struct _THE_CURRENT_ENVIRONMENT_RETURNED_VOID_FOR_THIS_QUERY;

struct _CCCL_TYPE_VISIBILITY_DEFAULT read_env_t
{
private:
  template <class _Rcvr, class _Query>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _Rcvr __rcvr_;

    _CCCL_API constexpr explicit __opstate_t(_Rcvr __rcvr) noexcept
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API void start() noexcept
    {
      // If the query invocation is noexcept, call it directly. Otherwise,
      // wrap it in a try-catch block and forward the exception to the
      // receiver.
      if constexpr (__nothrow_callable<_Query, env_of_t<_Rcvr>>)
      {
        // This looks like a use after move, but `set_value` takes its
        // arguments by forwarding reference, so it's safe.
        execution::set_value(static_cast<_Rcvr&&>(__rcvr_), _Query{}(execution::get_env(__rcvr_)));
      }
      else
      {
        _CCCL_TRY
        {
          execution::set_value(static_cast<_Rcvr&&>(__rcvr_), _Query{}(execution::get_env(__rcvr_)));
        }
        _CCCL_CATCH_ALL
        {
          execution::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
        }
      }
    }
  };

  struct __attrs_t
  {
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t) const noexcept
    {
      return completion_behavior::inline_completion;
    }
  };

public:
  template <class _Query>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  /// @brief Returns a sender that, when connected to a receiver and started,
  /// invokes the query with the receiver's environment and forwards the result
  /// to the receiver's `set_value` member.
  template <class _Query>
  _CCCL_NODEBUG_API constexpr __sndr_t<_Query> operator()(_Query) const noexcept;
};

template <class _Query>
struct _CCCL_TYPE_VISIBILITY_DEFAULT read_env_t::__sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    if constexpr (!__callable<_Query, _Env>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, read_env_t),
                                          _WHAT(_THE_CURRENT_ENVIRONMENT_LACKS_THIS_QUERY),
                                          _WITH_QUERY(_Query),
                                          _WITH_ENVIRONMENT(_Env)>();
    }
    else if constexpr (::cuda::std::is_void_v<__call_result_t<_Query, _Env>>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, read_env_t),
                                          _WHAT(_THE_CURRENT_ENVIRONMENT_RETURNED_VOID_FOR_THIS_QUERY),
                                          _WITH_QUERY(_Query),
                                          _WITH_ENVIRONMENT(_Env)>();
    }
    else
    {
      return completion_signatures<set_value_t(__call_result_t<_Query, _Env>)>{}
           + __eptr_completion_if<!__nothrow_callable<_Query, _Env>>();
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const noexcept -> __opstate_t<_Rcvr, _Query>
  {
    return __opstate_t<_Rcvr, _Query>{static_cast<_Rcvr&&>(__rcvr)};
  }

  [[nodiscard]] _CCCL_API static constexpr auto get_env() noexcept
  {
    return __attrs_t{};
  }

  _CCCL_NO_UNIQUE_ADDRESS read_env_t __tag;
  _CCCL_NO_UNIQUE_ADDRESS _Query __query;
};

template <class _Query>
_CCCL_NODEBUG_API constexpr read_env_t::__sndr_t<_Query> read_env_t::operator()(_Query __query) const noexcept
{
  return __sndr_t<_Query>{{}, __query};
}

template <class _Query>
inline constexpr size_t structured_binding_size<read_env_t::__sndr_t<_Query>> = 2;

_CCCL_GLOBAL_CONSTANT read_env_t read_env{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_READ_ENV
