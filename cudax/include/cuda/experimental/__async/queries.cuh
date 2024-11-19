//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/stop_token.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
template <class _Ty, class _Query>
auto __query_result_() -> decltype(__declval<_Ty>().query(_Query()));

template <class _Ty, class _Query>
using __query_result_t = decltype(__query_result_<_Ty, _Query>());

template <class _Ty, class _Query>
inline constexpr bool __queryable = __type_valid_v<__query_result_t, _Ty, _Query>;

#if defined(__CUDA_ARCH__)
template <class _Ty, class _Query>
inline constexpr bool __nothrow_queryable = true;
#else
template <class _Ty, class _Query>
using __nothrow_queryable_ = _CUDA_VSTD::enable_if_t<noexcept(__declval<_Ty>().query(_Query()))>;

template <class _Ty, class _Query>
inline constexpr bool __nothrow_queryable = __type_valid_v<__nothrow_queryable_, _Ty, _Query>;
#endif

_CCCL_GLOBAL_CONSTANT struct get_allocator_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept //
    -> decltype(__env.query(*this))
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  _CUDAX_API auto operator()(__ignore) const noexcept -> _CUDA_VSTD::allocator<void>
  {
    return {};
  }
} get_allocator{};

_CCCL_GLOBAL_CONSTANT struct get_stop_token_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept //
    -> decltype(__env.query(*this))
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  _CUDAX_API auto operator()(__ignore) const noexcept -> never_stop_token
  {
    return {};
  }
} get_stop_token{};

template <class _Ty>
using stop_token_of_t = __decay_t<__call_result_t<get_stop_token_t, _Ty>>;

template <class _Tag>
struct get_completion_scheduler_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept //
    -> decltype(__env.query(*this))
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }
};

template <class _Tag>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<_Tag> get_completion_scheduler{};

_CCCL_GLOBAL_CONSTANT struct get_scheduler_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept //
    -> decltype(__env.query(*this))
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }
} get_scheduler{};

_CCCL_GLOBAL_CONSTANT struct get_delegation_scheduler_t
{
  template <class _Env>
  _CUDAX_API auto operator()(const _Env& __env) const noexcept //
    -> decltype(__env.query(*this))
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }
} get_delegation_scheduler{};

enum class forward_progress_guarantee
{
  concurrent,
  parallel,
  weakly_parallel
};

_CCCL_GLOBAL_CONSTANT struct get_forward_progress_guarantee_t
{
  template <class _Sch>
  _CUDAX_API auto operator()(const _Sch& __sch) const noexcept //
    -> decltype(__async::__decay_copy(__sch.query(*this)))
  {
    static_assert(noexcept(__sch.query(*this)));
    return __sch.query(*this);
  }

  _CUDAX_API auto operator()(__ignore) const noexcept -> forward_progress_guarantee
  {
    return forward_progress_guarantee::weakly_parallel;
  }
} get_forward_progress_guarantee{};

_CCCL_GLOBAL_CONSTANT struct get_domain_t
{
  template <class _Sch>
  _CUDAX_API constexpr auto operator()(const _Sch& __sch) const noexcept //
    -> decltype(__async::__decay_copy(__sch.query(*this)))
  {
    return {};
  }
} get_domain{};

template <class _Sch>
using domain_of_t = __call_result_t<get_domain_t, _Sch>;

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
