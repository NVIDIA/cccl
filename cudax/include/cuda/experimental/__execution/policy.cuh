//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/std/__type_traits/is_convertible.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{

enum class execution_policy
{
  invalid_execution_policy,
  sequenced_host,
  sequenced_device,
  parallel_host,
  parallel_device,
  parallel_unsequenced_host,
  parallel_unsequenced_device,
  unsequenced_host,
  unsequenced_device,
};

_CCCL_GLOBAL_CONSTANT execution_policy seq_host         = execution_policy::sequenced_host;
_CCCL_GLOBAL_CONSTANT execution_policy seq_device       = execution_policy::sequenced_device;
_CCCL_GLOBAL_CONSTANT execution_policy par_host         = execution_policy::parallel_host;
_CCCL_GLOBAL_CONSTANT execution_policy par_device       = execution_policy::parallel_device;
_CCCL_GLOBAL_CONSTANT execution_policy par_unseq_host   = execution_policy::parallel_unsequenced_host;
_CCCL_GLOBAL_CONSTANT execution_policy par_unseq_device = execution_policy::parallel_unsequenced_device;
_CCCL_GLOBAL_CONSTANT execution_policy unseq_host       = execution_policy::unsequenced_host;
_CCCL_GLOBAL_CONSTANT execution_policy unseq_device     = execution_policy::unsequenced_device;

template <execution_policy _Policy>
inline constexpr bool __is_parallel_execution_policy =
  _Policy == execution_policy::parallel_host || _Policy == execution_policy::parallel_device
  || _Policy == execution_policy::parallel_unsequenced_host || _Policy == execution_policy::parallel_unsequenced_device;

template <execution_policy _Policy>
inline constexpr bool __is_unsequenced_execution_policy =
  _Policy == execution_policy::unsequenced_host || _Policy == execution_policy::unsequenced_device
  || _Policy == execution_policy::parallel_unsequenced_host || _Policy == execution_policy::parallel_unsequenced_device;

struct get_execution_policy_t;

template <class _Tp>
_CCCL_CONCEPT __has_member_get_execution_policy = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, decltype(__t.get_execution_policy()), execution_policy)));

template <class _Env>
_CCCL_CONCEPT __has_query_get_execution_policy = _CCCL_REQUIRES_EXPR((_Env))(
  requires(!__has_member_get_execution_policy<_Env>),
  requires(_CCCL_TRAIT(_CUDA_VSTD::is_convertible,
                       _CUDA_STD_EXEC::__query_result_t<const _Env&, get_execution_policy_t>,
                       execution_policy)));

struct get_execution_policy_t
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_execution_policy<_Tp>)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI execution_policy operator()(const _Tp& __t) const noexcept
  {
    return __t.get_execution_policy();
  }

  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__has_query_get_execution_policy<_Env>)
  [[nodiscard]] _CCCL_HIDE_FROM_ABI execution_policy operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_execution_policy_t get_execution_policy{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif //__CUDAX___EXECUTION_POLICY_CUH
