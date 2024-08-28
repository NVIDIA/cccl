//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_QUERIES_H
#define __CUDAX_ASYNC_DETAIL_QUERIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/allocator.h>

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/stop_token.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
template <class Ty, class Query>
auto _query_result_() -> decltype(DECLVAL(Ty).query(Query()));

template <class Ty, class Query>
using _query_result_t = decltype(_query_result_<Ty, Query>());

template <class Ty, class Query>
_CCCL_INLINE_VAR constexpr bool _queryable = _mvalid_q<_query_result_t, Ty, Query>;

#if defined(__CUDA_ARCH__)
template <class Ty, class Query>
_CCCL_INLINE_VAR constexpr bool _nothrow_queryable = true;
#else
template <class Ty, class Query>
using _nothrow_queryable_ = _mif<noexcept(DECLVAL(Ty).query(Query()))>;

template <class Ty, class Query>
_CCCL_INLINE_VAR constexpr bool _nothrow_queryable = _mvalid_q<_nothrow_queryable_, Ty, Query>;
#endif

_CCCL_GLOBAL_CONSTANT struct get_allocator_t
{
  template <class Env>
  _CCCL_HOST_DEVICE auto operator()(const Env& env) const noexcept //
    -> decltype(env.query(*this))
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }

  _CCCL_HOST_DEVICE auto operator()(_ignore) const noexcept -> _CUDA_VSTD::allocator<void>
  {
    return {};
  }
} get_allocator{};

_CCCL_GLOBAL_CONSTANT struct get_stop_token_t
{
  template <class Env>
  _CCCL_HOST_DEVICE auto operator()(const Env& env) const noexcept //
    -> decltype(env.query(*this))
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }

  _CCCL_HOST_DEVICE auto operator()(_ignore) const noexcept -> never_stop_token
  {
    return {};
  }
} get_stop_token{};

template <class T>
using stop_token_of_t = _decay_t<_call_result_t<get_stop_token_t, T>>;

template <class Tag>
struct get_completion_scheduler_t
{
  template <class Env>
  _CCCL_HOST_DEVICE auto operator()(const Env& env) const noexcept //
    -> decltype(env.query(*this))
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }
};

template <class Tag>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<Tag> get_completion_scheduler{};

_CCCL_GLOBAL_CONSTANT struct get_scheduler_t
{
  template <class Env>
  _CCCL_HOST_DEVICE auto operator()(const Env& env) const noexcept //
    -> decltype(env.query(*this))
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }
} get_scheduler{};

_CCCL_GLOBAL_CONSTANT struct get_delegatee_scheduler_t
{
  template <class Env>
  _CCCL_HOST_DEVICE auto operator()(const Env& env) const noexcept //
    -> decltype(env.query(*this))
  {
    static_assert(noexcept(env.query(*this)));
    return env.query(*this);
  }
} get_delegatee_scheduler{};

enum class forward_progress_guarantee
{
  concurrent,
  parallel,
  weakly_parallel
};

_CCCL_GLOBAL_CONSTANT struct get_forward_progress_guarantee_t
{
  template <class Sch>
  _CCCL_HOST_DEVICE auto operator()(const Sch& sch) const noexcept //
    -> decltype(__async::_decay_copy(sch.query(*this)))
  {
    static_assert(noexcept(sch.query(*this)));
    return sch.query(*this);
  }

  _CCCL_HOST_DEVICE auto operator()(_ignore) const noexcept -> forward_progress_guarantee
  {
    return forward_progress_guarantee::weakly_parallel;
  }
} get_forward_progress_guarantee{};

_CCCL_GLOBAL_CONSTANT struct get_domain_t
{
  template <class Sch>
  _CCCL_HOST_DEVICE constexpr auto operator()(const Sch& sch) const noexcept //
    -> decltype(__async::_decay_copy(sch.query(*this)))
  {
    return {};
  }
} get_domain{};

template <class Sch>
using domain_of_t = _call_result_t<get_domain_t, Sch>;

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
