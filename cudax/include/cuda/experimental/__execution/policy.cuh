//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX___EXECUTION_POLICY_CUH
#define __CUDAX___EXECUTION_POLICY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__execution/policy.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_execution_policy.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{

using ::cuda::std::execution::__execution_policy;
using ::cuda::std::execution::par;
using ::cuda::std::execution::par_unseq;
using ::cuda::std::execution::seq;
using ::cuda::std::execution::unseq;

struct any_execution_policy
{
  using type       = any_execution_policy;
  using value_type = __execution_policy;

  _CCCL_HIDE_FROM_ABI any_execution_policy() = default;

  template <__execution_policy _Policy>
  _CCCL_HOST_API constexpr any_execution_policy(::cuda::std::execution::__policy<_Policy>) noexcept
      : value(_Policy)
  {}

  _CCCL_HOST_API constexpr operator __execution_policy() const noexcept
  {
    return value;
  }

  _CCCL_HOST_API constexpr auto operator()() const noexcept -> __execution_policy
  {
    return value;
  }

  template <__execution_policy _Policy>
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(const any_execution_policy& pol, const ::cuda::std::execution::__policy<_Policy>&) noexcept
  {
    return pol.value == _Policy;
  }

#if _CCCL_STD_VER <= 2017
  template <__execution_policy _Policy>
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator==(const ::cuda::std::execution::__policy<_Policy>&, const any_execution_policy& pol) noexcept
  {
    return pol.value == _Policy;
  }

  template <__execution_policy _Policy>
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(const any_execution_policy& pol, const ::cuda::std::execution::__policy<_Policy>&) noexcept
  {
    return pol.value != _Policy;
  }

  template <__execution_policy _Policy>
  [[nodiscard]] _CCCL_HOST_API friend constexpr bool
  operator!=(const ::cuda::std::execution::__policy<_Policy>&, const any_execution_policy& pol)
  {
    return pol.value != _Policy;
  }
#endif // _CCCL_STD_VER <= 2017

  __execution_policy value = __execution_policy::__invalid_execution_policy;
};

struct get_execution_policy_t;

template <class _Tp>
_CCCL_CONCEPT __has_member_get_execution_policy = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(::cuda::std::is_convertible_v<decltype(__t.get_execution_policy()), __execution_policy>));

template <class _Env>
_CCCL_CONCEPT __has_query_get_execution_policy = _CCCL_REQUIRES_EXPR((_Env))(
  requires(!__has_member_get_execution_policy<_Env>),
  requires(::cuda::std::is_convertible_v<::cuda::std::execution::__query_result_t<const _Env&, get_execution_policy_t>,
                                         __execution_policy>));

struct get_execution_policy_t
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_execution_policy<_Tp>)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI auto operator()(const _Tp& __t) const noexcept
  {
    return __t.get_execution_policy();
  }

  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__has_query_get_execution_policy<_Env>)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_execution_policy_t get_execution_policy{};

} // namespace cuda::experimental::execution

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <>
inline constexpr bool is_execution_policy_v<::cuda::experimental::execution::any_execution_policy> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/experimental/__execution/epilogue.cuh>

#endif //__CUDAX___EXECUTION_POLICY_CUH
