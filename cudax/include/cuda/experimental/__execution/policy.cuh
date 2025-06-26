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
#include <cuda/std/__type_traits/is_convertible.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
namespace execution
{
struct sequenced_policy;
struct parallel_policy;
struct parallel_unsequenced_policy;
struct unsequenced_policy;
struct any_execution_policy;
} // namespace execution

// execution policy type trait
template <class _Ty>
inline constexpr bool is_execution_policy_v = false;

template <>
inline constexpr bool is_execution_policy_v<execution::sequenced_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<execution::parallel_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<execution::parallel_unsequenced_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<execution::unsequenced_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<execution::any_execution_policy> = true;

template <class _Ty>
struct is_execution_policy : _CUDA_VSTD::bool_constant<is_execution_policy_v<_Ty>>
{};

namespace execution
{
enum class __execution_policy
{
  invalid_execution_policy,
  sequenced,
  parallel,
  parallel_unsequenced,
  unsequenced,
};

template <__execution_policy _Policy>
struct __policy;

struct any_execution_policy
{
  using type       = any_execution_policy;
  using value_type = __execution_policy;

  _CCCL_HIDE_FROM_ABI any_execution_policy() = default;

  template <__execution_policy _Policy>
  _CCCL_HOST_API constexpr any_execution_policy(__policy<_Policy> __pol) noexcept
      : value(__pol)
  {}

  _CCCL_HOST_API constexpr operator __execution_policy() const noexcept
  {
    return value;
  }

  _CCCL_HOST_API constexpr auto operator()() const noexcept -> __execution_policy
  {
    return value;
  }

  __execution_policy value = __execution_policy::invalid_execution_policy;
};

template <__execution_policy _Policy>
struct _CCCL_DECLSPEC_EMPTY_BASES __policy : _CUDA_VSTD::integral_constant<__execution_policy, _Policy>
{};

struct sequenced_policy : __policy<__execution_policy::sequenced>
{};

struct parallel_policy : __policy<__execution_policy::parallel>
{};

struct parallel_unsequenced_policy : __policy<__execution_policy::parallel_unsequenced>
{};

struct unsequenced_policy : __policy<__execution_policy::unsequenced>
{};

_CCCL_GLOBAL_CONSTANT sequenced_policy seq{};
_CCCL_GLOBAL_CONSTANT parallel_policy par{};
_CCCL_GLOBAL_CONSTANT parallel_unsequenced_policy par_unseq{};
_CCCL_GLOBAL_CONSTANT unsequenced_policy unseq{};

template <__execution_policy _Policy>
inline constexpr bool __is_parallel_execution_policy =
  _Policy == __execution_policy::parallel || _Policy == __execution_policy::parallel_unsequenced;

template <__execution_policy _Policy>
inline constexpr bool __is_unsequenced_execution_policy =
  _Policy == __execution_policy::unsequenced || _Policy == __execution_policy::parallel_unsequenced;

struct get_execution_policy_t;

template <class _Tp>
_CCCL_CONCEPT __has_member_get_execution_policy = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, decltype(__t.get_execution_policy()), __execution_policy)));

template <class _Env>
_CCCL_CONCEPT __has_query_get_execution_policy = _CCCL_REQUIRES_EXPR((_Env))(
  requires(!__has_member_get_execution_policy<_Env>),
  requires(_CCCL_TRAIT(_CUDA_VSTD::is_convertible,
                       _CUDA_STD_EXEC::__query_result_t<const _Env&, get_execution_policy_t>,
                       __execution_policy)));

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

} // namespace execution
} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif //__CUDAX___EXECUTION_POLICY_CUH
