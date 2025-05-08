//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_QUERIES
#define __CUDAX_ASYNC_DETAIL_QUERIES

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

#include <cuda/std/__execution/env.h>

#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
using _CUDA_STD_EXEC::__queryable_with;

//////////////////////////////////////////////////////////////////////////////////////////
// get_allocator
_CCCL_GLOBAL_CONSTANT struct get_allocator_t
{
  template <class _Env>
  _CCCL_API auto operator()(const _Env& __env) const noexcept
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
} get_allocator{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_stop_token
_CCCL_GLOBAL_CONSTANT struct get_stop_token_t
{
  template <class _Env>
  _CCCL_API auto operator()(const _Env& __env) const noexcept
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
} get_stop_token{};

template <class _Ty>
using stop_token_of_t _CCCL_NODEBUG_ALIAS = __decay_t<__call_result_t<get_stop_token_t, _Ty>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_scheduler
template <class _Tag>
struct get_completion_scheduler_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_completion_scheduler_t>)
  _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }
};

template <class _Tag>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<_Tag> get_completion_scheduler{};

template <class _Env, class _Tag = set_value_t>
using __completion_scheduler_of_t _CCCL_NODEBUG_ALIAS =
  __decay_t<__call_result_t<get_completion_scheduler_t<_Tag>, _Env>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_scheduler
_CCCL_GLOBAL_CONSTANT struct get_scheduler_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_scheduler_t>)
  _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }
} get_scheduler{};

template <class _Env>
using __scheduler_of_t _CCCL_NODEBUG_ALIAS = __decay_t<__call_result_t<get_scheduler_t, _Env>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_delegation_scheduler
_CCCL_GLOBAL_CONSTANT struct get_delegation_scheduler_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_delegation_scheduler_t>)
  _CCCL_API auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
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

_CCCL_GLOBAL_CONSTANT struct get_forward_progress_guarantee_t
{
  template <class _Sch>
  _CCCL_API auto operator()(const _Sch& __sch) const noexcept
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
// get_domain
_CCCL_GLOBAL_CONSTANT struct get_domain_t
{
  template <class _Env>
  _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    if constexpr (__queryable_with<_Env, get_domain_t>)
    {
      static_assert(noexcept(__env.query(*this)));
      return __env.query(*this);
    }
    else
    {
      return default_domain{};
    }
  }
} get_domain{};

template <class _Env>
using __domain_of_t _CCCL_NODEBUG_ALIAS = __call_result_t<get_domain_t, _Env>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif
