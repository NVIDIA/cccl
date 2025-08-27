//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_EXECUTION_POLICY_H
#define _CUDA_STD___TYPE_TRAITS_IS_EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/policy.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/void_t.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class>
inline constexpr bool is_execution_policy_v = false;

// Ensure we ignore cv qualifiers
template <class _Tp>
inline constexpr bool is_execution_policy_v<const _Tp> = is_execution_policy_v<_Tp>;

template <class _Tp>
inline constexpr bool is_execution_policy_v<volatile _Tp> = is_execution_policy_v<_Tp>;

template <class _Tp>
inline constexpr bool is_execution_policy_v<const volatile _Tp> = is_execution_policy_v<_Tp>;

// Explicitly mark our execution policies as such
template <>
inline constexpr bool is_execution_policy_v<::cuda::std::execution::sequenced_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<::cuda::std::execution::parallel_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<::cuda::std::execution::parallel_unsequenced_policy> = true;

template <>
inline constexpr bool is_execution_policy_v<::cuda::std::execution::unsequenced_policy> = true;

template <class _Tp>
struct _CCCL_NO_SPECIALIZATIONS is_execution_policy : bool_constant<is_execution_policy_v<_Tp>>
{};

// Detect parallel policies
template <class, class = void>
inline constexpr bool __is_parallel_execution_policy_v = false;

template <class _Policy>
inline constexpr bool __is_parallel_execution_policy_v<_Policy, void_t<decltype(_Policy::__policy_)>> =
  __satisfies_execution_policy(_Policy::__policy_, ::cuda::std::execution::__execution_policy::__parallel);

// Detect unsequenced policies
template <class, class = void>
inline constexpr bool __is_unsequenced_execution_policy_v = false;

template <class _Policy>
inline constexpr bool __is_unsequenced_execution_policy_v<_Policy, void_t<decltype(_Policy::__policy_)>> =
  __satisfies_execution_policy(_Policy::__policy_, ::cuda::std::execution::__execution_policy::__unsequenced);

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_EXECUTION_POLICY_H
