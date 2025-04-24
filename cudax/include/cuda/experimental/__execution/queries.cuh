//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_QUERIES
#define __CUDAX_EXECUTION_QUERIES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>

_CCCL_SUPPRESS_DEPRECATED_PUSH
#include <cuda/std/__memory/allocator.h>
_CCCL_SUPPRESS_DEPRECATED_POP

#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__execution/env.h>

#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__launch/configuration.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// NOLINTBEGIN(misc-unused-using-decls)
using _CUDA_STD_EXEC::__forwarding_query;
using _CUDA_STD_EXEC::__unwrap_reference_t;
using _CUDA_STD_EXEC::env;
using _CUDA_STD_EXEC::env_of_t;
using _CUDA_STD_EXEC::forwarding_query;
using _CUDA_STD_EXEC::forwarding_query_t;
using _CUDA_STD_EXEC::get_env;
using _CUDA_STD_EXEC::get_env_t;
using _CUDA_STD_EXEC::prop;

using _CUDA_STD_EXEC::__nothrow_queryable_with;
using _CUDA_STD_EXEC::__query_result_t;
using _CUDA_STD_EXEC::__queryable_with;
// NOLINTEND(misc-unused-using-decls)

//////////////////////////////////////////////////////////////////////////////////////////
// get_allocator
_CCCL_GLOBAL_CONSTANT struct get_allocator_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    if constexpr (__queryable_with<_Env, get_allocator_t>)
    {
      static_assert(noexcept(__env.query(*this)));
      return __env.query(*this);
    }
    else
    {
      return _CUDA_VSTD::allocator<void>{};
    }
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_allocator{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_stop_token
_CCCL_GLOBAL_CONSTANT struct get_stop_token_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    if constexpr (__queryable_with<_Env, get_stop_token_t>)
    {
      static_assert(noexcept(__env.query(*this)));
      return __env.query(*this);
    }
    else
    {
      return never_stop_token{};
    }
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_stop_token{};

template <class _Ty>
using stop_token_of_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::decay_t<_CUDA_VSTD::__call_result_t<get_stop_token_t, _Ty>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_scheduler
template <class _Tag>
struct get_completion_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_completion_scheduler_t>)
  [[nodiscard]] _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

template <class _Tag>
extern _CUDA_VSTD::__undefined<_Tag> get_completion_scheduler;

// Explicitly instantiate these because of variable template weirdness in device code
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_value_t> get_completion_scheduler<set_value_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_error_t> get_completion_scheduler<set_error_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_stopped_t> get_completion_scheduler<set_stopped_t>{};

template <class _Env, class _Tag = set_value_t>
using __completion_scheduler_of_t _CCCL_NODEBUG_ALIAS =
  _CUDA_VSTD::decay_t<_CUDA_VSTD::__call_result_t<get_completion_scheduler_t<_Tag>, _Env>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_scheduler
_CCCL_GLOBAL_CONSTANT struct get_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_scheduler_t>)
  [[nodiscard]] _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_scheduler{};

template <class _Env>
using __scheduler_of_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::decay_t<_CUDA_VSTD::__call_result_t<get_scheduler_t, _Env>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_delegation_scheduler
_CCCL_GLOBAL_CONSTANT struct get_delegation_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_delegation_scheduler_t>)
  [[nodiscard]] _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_delegation_scheduler{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_forward_progress_guarantee
enum class forward_progress_guarantee
{
  concurrent,
  parallel,
  weakly_parallel
};

// This query is not a forwarding query.
_CCCL_GLOBAL_CONSTANT struct get_forward_progress_guarantee_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sch>
  [[nodiscard]] _CCCL_API auto operator()(const _Sch& __sch) const noexcept
  {
    if constexpr (__queryable_with<_Sch, get_forward_progress_guarantee_t>)
    {
      static_assert(noexcept(__sch.query(*this)));
      return __sch.query(*this);
    }
    else
    {
      return forward_progress_guarantee::weakly_parallel;
    }
  }
} get_forward_progress_guarantee{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_launch_config: A sender can define this attribute to control the launch configuration
// of the kernel it will launch when executed on a CUDA stream scheduler.
_CCCL_GLOBAL_CONSTANT struct get_launch_config_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Env>
  [[nodiscard]] _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    if constexpr (__queryable_with<_Env, get_launch_config_t>)
    {
      static_assert(noexcept(__env.query(*this)), "The get_launch_config query must be noexcept.");
      static_assert(__is_specialization_of_v<decltype(__env.query(*this)), kernel_config>,
                    "The get_launch_config query must return a kernel_config type.");
      return __env.query(*this);
    }
    else
    {
      return experimental::make_config(grid_dims<1>, block_dims<1>);
    }
  }
} get_launch_config{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_QUERIES
