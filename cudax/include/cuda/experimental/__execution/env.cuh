//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX___EXECUTION_ENV_CUH
#define __CUDAX___EXECUTION_ENV_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/device_memory_resource.cuh>
#include <cuda/experimental/__memory_resource/get_memory_resource.cuh>
#include <cuda/experimental/__stream/get_stream.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
namespace execution
{
using _CUDA_STD_EXEC::env;
using _CUDA_STD_EXEC::env_of_t;
using _CUDA_STD_EXEC::get_env;
using _CUDA_STD_EXEC::prop;

using _CUDA_STD_EXEC::__nothrow_queryable_with;
using _CUDA_STD_EXEC::__query_result_t;
using _CUDA_STD_EXEC::__queryable_with;
} // namespace execution

template <class... _Properties>
class env_t
{
private:
  using __resource   = any_async_resource<_Properties...>;
  using __stream_ref = stream_ref;

  __resource __mr_                      = device_memory_resource{};
  __stream_ref __stream_                = detail::__invalid_stream;
  execution::execution_policy __policy_ = execution::execution_policy::invalid_execution_policy;

public:
  //! @brief Default constructs an environment using ``device_memory_resource`` as the resource the default stream
  //! ``execution_policy::invalid_execution_policy`` as the execution policy
  _CCCL_HIDE_FROM_ABI env_t() = default;

  //! @brief Construct an env_t from an any_resource, a stream and a policy
  //! @param __mr The any_resource passed in
  //! @param __stream The stream_ref passed in
  //! @param __policy The execution_policy passed in
  _CCCL_HIDE_FROM_ABI
  env_t(__resource __mr,
        __stream_ref __stream                = detail::__invalid_stream,
        execution::execution_policy __policy = execution::execution_policy::invalid_execution_policy) noexcept
      : __mr_(_CUDA_VSTD::move(__mr))
      , __stream_(__stream)
      , __policy_(__policy)
  {}

  //! @brief Checks whether another env is compatible with this one. That requires it to have queries for the three
  //! properties we need
  template <class _Env>
  static constexpr bool __is_compatible_env =
    _CUDA_STD_EXEC::__queryable_with<_Env, get_memory_resource_t> //
    && _CUDA_STD_EXEC::__queryable_with<_Env, get_stream_t>
    && _CUDA_STD_EXEC::__queryable_with<_Env, execution::get_execution_policy_t>;

  //! @brief Construct from an environment that has the right queries
  //! @param __env The environment we are querying for the required information
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES((!_CCCL_TRAIT(_CUDA_VSTD::is_same, _Env, env_t)) _CCCL_AND __is_compatible_env<_Env>)
  _CCCL_HIDE_FROM_ABI env_t(const _Env& __env) noexcept
      : __mr_(__env.query(get_memory_resource))
      , __stream_(__env.query(get_stream))
      , __policy_(__env.query(execution::get_execution_policy))
  {}

  [[nodiscard]] _CCCL_HIDE_FROM_ABI const __resource& query(get_memory_resource_t) const noexcept
  {
    return __mr_;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI __stream_ref query(get_stream_t) const noexcept
  {
    return __stream_;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI execution::execution_policy query(execution::get_execution_policy_t) const noexcept
  {
    return __policy_;
  }
};

} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif //__CUDAX___EXECUTION_ENV_CUH
