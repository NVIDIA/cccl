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

#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__memory_resource/any_resource.cuh>
#include <cuda/experimental/__memory_resource/device_memory_resource.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
namespace execution
{
template <class _Env, class _Query>
_CCCL_CONCEPT __statically_queryable_with = //
  _CCCL_REQUIRES_EXPR((_Env, _Query)) //
  ( //
    (_CUDA_VSTD::remove_cvref_t<_Env>::query(_Query{})) //
  );

template <class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __fwd_env_;

//! \brief __env_ref_ is a utility that builds a queryable object from a reference
//! to another queryable object.
template <class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_ref_
{
  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__queryable_with<_Env, _Query>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
    -> __query_result_t<_Env, _Query>
  {
    return __env_.query(_Query{});
  }

  _Env const& __env_;
};

namespace __detail
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_ref_fn
{
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(env<>) const noexcept -> env<>
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Env, class = _Env*) // not considered if _Env is a reference type
  _CCCL_REQUIRES((!__is_specialization_of_v<_Env, __fwd_env_>) )
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Env&& __env) const noexcept -> _Env
  {
    return static_cast<_Env&&>(__env);
  }

  template <class _Env>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(const _Env& __env) const noexcept -> __env_ref_<_Env>
  {
    return __env_ref_<_Env>{__env};
  }

  template <class _Env>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(__env_ref_<_Env> __env) const noexcept -> __env_ref_<_Env>
  {
    return __env;
  }

  template <class _Env>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(const __fwd_env_<_Env>& __env) const noexcept
    -> __fwd_env_<_Env const&>
  {
    return __fwd_env_<_Env const&>{__env.__env_};
  }
};
} // namespace __detail

template <class _Env>
using __env_ref_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<__detail::__env_ref_fn, _Env>;

_CCCL_GLOBAL_CONSTANT __detail::__env_ref_fn __env_ref{};

//! \brief __fwd_env_ is a utility that forwards queries to a given queryable object
//! provided those queries that satisfy the __forwarding_query concept.
template <class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __fwd_env_
{
  _CCCL_TEMPLATE(class _Query)
  _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<_Env, _Query>)
  [[nodiscard]] _CCCL_API constexpr auto query(_Query) const noexcept(__nothrow_queryable_with<_Env, _Query>)
    -> __query_result_t<_Env, _Query>
  {
    return __env_.query(_Query{});
  }

  _Env __env_;
};

namespace __detail
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT __fwd_env_fn
{
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(env<>) const noexcept -> env<>
  {
    return {};
  }

  template <class _Env>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(__env_ref_<_Env> __env) const noexcept
    -> __fwd_env_<_Env const&>
  {
    return __fwd_env_<_Env const&>{__env.__env_};
  }

  template <class _Env>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(_Env&& __env) const noexcept(__nothrow_movable<_Env>)
    -> decltype(auto)
  {
    if constexpr (__is_specialization_of_v<_CUDA_VSTD::remove_cvref_t<_Env>, __fwd_env_>)
    {
      // If the environment is already a forwarding environment, we can just return it.
      return static_cast<_Env>(static_cast<_Env&&>(__env)); // take care to not return an rvalue reference
    }
    else
    {
      return __fwd_env_<_Env>{static_cast<_Env&&>(__env)};
    }
  }
};
} // namespace __detail

template <class _Env>
using __fwd_env_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__call_result_t<__detail::__fwd_env_fn, _Env>;

_CCCL_GLOBAL_CONSTANT __detail::__fwd_env_fn __fwd_env{};

//! \brief __sch_env_t is a utility that builds a queryable object from a scheduler. It
//! defines the `get_scheduler` query and provides a default for the `get_domain` query.
template <class _Sch>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sch_env_t
{
  [[nodiscard]] _CCCL_API constexpr auto query(get_scheduler_t) const noexcept -> _Sch
  {
    return __sch_;
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t) noexcept
  {
    return __query_result_or_t<_Sch, get_domain_t, default_domain>{};
  }

  _Sch __sch_;
};

template <class _Sch>
_CCCL_HOST_DEVICE __sch_env_t(_Sch) -> __sch_env_t<_Sch>;

} // namespace execution

template <class... _Properties>
class env_t
{
private:
  using __resource   = any_async_resource<_Properties...>;
  using __stream_ref = stream_ref;

  __resource __mr_;
  __stream_ref __stream_                    = __detail::__invalid_stream;
  execution::any_execution_policy __policy_ = {};

public:
  //! @brief Construct an env_t from an any_resource, a stream and a policy
  //! @param __mr The any_resource passed in
  //! @param __stream The stream_ref passed in
  //! @param __policy The execution_policy passed in
  _CCCL_HIDE_FROM_ABI env_t(__resource __mr,
                            __stream_ref __stream                    = __detail::__invalid_stream,
                            execution::any_execution_policy __policy = {}) noexcept
      : __mr_(_CUDA_VSTD::move(__mr))
      , __stream_(__stream)
      , __policy_(__policy)
  {}

  //! @brief Checks whether another env is compatible with this one. That requires it to have queries for the three
  //! properties we need
  template <class _Env>
  static constexpr bool __is_compatible_env =
    _CUDA_STD_EXEC::__queryable_with<_Env, ::cuda::mr::get_memory_resource_t> //
    && _CUDA_STD_EXEC::__queryable_with<_Env, ::cuda::get_stream_t>
    && _CUDA_STD_EXEC::__queryable_with<_Env, execution::get_execution_policy_t>;

  //! @brief Construct from an environment that has the right queries
  //! @param __env The environment we are querying for the required information
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES((!_CCCL_TRAIT(_CUDA_VSTD::is_same, _Env, env_t)) _CCCL_AND __is_compatible_env<_Env>)
  _CCCL_HIDE_FROM_ABI env_t(const _Env& __env) noexcept
      : __mr_(__env.query(::cuda::mr::get_memory_resource))
      , __stream_(__env.query(::cuda::get_stream))
      , __policy_(__env.query(execution::get_execution_policy))
  {}

  [[nodiscard]] _CCCL_HIDE_FROM_ABI const __resource& query(::cuda::mr::get_memory_resource_t) const noexcept
  {
    return __mr_;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI __stream_ref query(::cuda::get_stream_t) const noexcept
  {
    return __stream_;
  }

  [[nodiscard]] _CCCL_HIDE_FROM_ABI execution::any_execution_policy
  query(execution::get_execution_policy_t) const noexcept
  {
    return __policy_;
  }
};

} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif //__CUDAX___EXECUTION_ENV_CUH
