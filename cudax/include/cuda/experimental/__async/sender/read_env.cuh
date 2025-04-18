//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_READ_ENV
#define __CUDAX_ASYNC_DETAIL_READ_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_void.h>

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/env.cuh>
#include <cuda/experimental/__async/sender/exception.cuh>
#include <cuda/experimental/__async/sender/queries.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>
#include <cuda/experimental/__async/sender/visit.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
struct _THE_CURRENT_ENVIRONMENT_LACKS_THIS_QUERY;
struct _THE_CURRENT_ENVIRONMENT_RETURNED_VOID_FOR_THIS_QUERY;

struct read_env_t
{
private:
  template <class _Rcvr, class _Query>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _Rcvr __rcvr_;

    _CUDAX_API explicit __opstate_t(_Rcvr __rcvr)
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
    {}

    _CUDAX_IMMOVABLE(__opstate_t);

    _CUDAX_API void start() noexcept
    {
      // If the query invocation is noexcept, call it directly. Otherwise,
      // wrap it in a try-catch block and forward the exception to the
      // receiver.
      if constexpr (__nothrow_callable<_Query, env_of_t<_Rcvr>>)
      {
        // This looks like a use after move, but `set_value` takes its
        // arguments by forwarding reference, so it's safe.
        __async::set_value(static_cast<_Rcvr&&>(__rcvr_), _Query()(__async::get_env(__rcvr_)));
      }
      else
      {
        _CUDAX_TRY( //
          ({        //
            __async::set_value(static_cast<_Rcvr&&>(__rcvr_), _Query()(__async::get_env(__rcvr_)));
          }),
          _CUDAX_CATCH(...) //
          ({                //
            __async::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
          }) //
        )
      }
    }
  };

public:
  template <class _Query>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  /// @brief Returns a sender that, when connected to a receiver and started,
  /// invokes the query with the receiver's environment and forwards the result
  /// to the receiver's `set_value` member.
  template <class _Query>
  _CUDAX_TRIVIAL_API constexpr __sndr_t<_Query> operator()(_Query) const noexcept;
};

template <class _Query>
struct _CCCL_TYPE_VISIBILITY_DEFAULT read_env_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS read_env_t __tag;
  _CCCL_NO_UNIQUE_ADDRESS _Query __query;

  template <class _Self, class _Env>
  _CUDAX_API static constexpr auto get_completion_signatures()
  {
    if constexpr (!_CUDA_VSTD::__is_callable_v<_Query, _Env>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, read_env_t),
                                          _WHAT(_THE_CURRENT_ENVIRONMENT_LACKS_THIS_QUERY),
                                          _WITH_QUERY(_Query),
                                          _WITH_ENVIRONMENT(_Env)>();
    }
    else if constexpr (_CUDA_VSTD::is_void_v<__call_result_t<_Query, _Env>>)
    {
      return invalid_completion_signature<_WHERE(_IN_ALGORITHM, read_env_t),
                                          _WHAT(_THE_CURRENT_ENVIRONMENT_RETURNED_VOID_FOR_THIS_QUERY),
                                          _WITH_QUERY(_Query),
                                          _WITH_ENVIRONMENT(_Env)>();
    }
    else if constexpr (__nothrow_callable<_Query, _Env>)
    {
      return completion_signatures<set_value_t(__call_result_t<_Query, _Env>)>{};
    }
    else
    {
      return completion_signatures<set_value_t(__call_result_t<_Query, _Env>), set_error_t(::std::exception_ptr)>{};
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const noexcept(__nothrow_movable<_Rcvr>) -> __opstate_t<_Rcvr, _Query>
  {
    return __opstate_t<_Rcvr, _Query>{static_cast<_Rcvr&&>(__rcvr)};
  }
};

template <class _Query>
_CUDAX_TRIVIAL_API constexpr read_env_t::__sndr_t<_Query> read_env_t::operator()(_Query __query) const noexcept
{
  return __sndr_t<_Query>{{}, __query};
}

template <class _Query>
inline constexpr size_t structured_binding_size<read_env_t::__sndr_t<_Query>> = 2;

_CCCL_GLOBAL_CONSTANT read_env_t read_env{};

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
