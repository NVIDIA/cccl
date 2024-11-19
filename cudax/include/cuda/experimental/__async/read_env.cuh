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

#include <cuda/std/__type_traits/conditional.h>

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/env.cuh>
#include <cuda/experimental/__async/exception.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
struct THE_CURRENT_ENVIRONMENT_LACKS_THIS_QUERY;

struct read_env_t
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC
  template <class _Query, class _Env>
  using __error_env_lacks_query = //
    _ERROR<_WHERE(_IN_ALGORITHM, read_env_t),
           _WHAT(THE_CURRENT_ENVIRONMENT_LACKS_THIS_QUERY),
           _WITH_QUERY(_Query),
           _WITH_ENVIRONMENT(_Env)>;

  struct __completions_fn
  {
    template <class _Query, class _Env>
    using __call = _CUDA_VSTD::conditional_t<
      __nothrow_callable<_Query, _Env>,
      completion_signatures<set_value_t(__call_result_t<_Query, _Env>)>,
      completion_signatures<set_value_t(__call_result_t<_Query, _Env>), set_error_t(::std::exception_ptr)>>;
  };

  template <class _Rcvr, class _Query>
  struct __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = //
      _CUDA_VSTD::__type_call<_CUDA_VSTD::conditional_t<__callable<_Query, env_of_t<_Rcvr>>,
                                                        __completions_fn,
                                                        __error_env_lacks_query<_Query, env_of_t<_Rcvr>>>,
                              _Query,
                              env_of_t<_Rcvr>>;

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
          ( //
            { //
              __async::set_value(static_cast<_Rcvr&&>(__rcvr_), _Query()(__async::get_env(__rcvr_)));
            }),
          _CUDAX_CATCH(...)( //
            { //
              __async::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
            }))
      }
    }
  };

  // This makes read_env a dependent sender:
  template <class _Query>
  struct __opstate_t<receiver_archetype, _Query>
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = dependent_completions;
    _CUDAX_API explicit __opstate_t(receiver_archetype);
    _CUDAX_API void start() noexcept;
  };

  template <class _Query>
  struct __sndr_t;

public:
  /// @brief Returns a sender that, when connected to a receiver and started,
  /// invokes the query with the receiver's environment and forwards the result
  /// to the receiver's `set_value` member.
  template <class _Query>
  _CUDAX_TRIVIAL_API constexpr __sndr_t<_Query> operator()(_Query) const noexcept;
};

template <class _Query>
struct read_env_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS read_env_t __tag;
  _CCCL_NO_UNIQUE_ADDRESS _Query __query;

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

_CCCL_GLOBAL_CONSTANT read_env_t read_env{};

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
