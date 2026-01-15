//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_EXECUTION_POLICY_H
#define _CUDA_STD___FWD_EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

//! @brief Enumerates the standard execution policies
enum class __execution_policy : uint8_t
{
  __invalid_execution_policy = 0,
  __sequenced                = 1 << 0,
  __parallel                 = 1 << 1,
  __unsequenced              = 1 << 2,
  __parallel_unsequenced     = __execution_policy::__parallel | __execution_policy::__unsequenced,
};

//! @brief Extracts the execution policy from the stored _Policy
template <uint32_t _Policy>
inline constexpr __execution_policy __policy_to_execution_policy = __execution_policy{(_Policy & uint32_t{0x000000FF})};

//! @brief Enumerates the different backends we support
//! @note Not an enum class because a user might specify multiple backends
enum __execution_backend : uint8_t
{
  // The backends we provide
  __none = 0,
#if _CCCL_HAS_BACKEND_CUDA()
  __cuda = 1 << 1,
#endif // _CCCL_HAS_BACKEND_CUDA()
#if _CCCL_HAS_BACKEND_OMP()
  __omp = 1 << 2,
#endif // _CCCL_HAS_BACKEND_OMP()
#if _CCCL_HAS_BACKEND_TBB()
  __tbb = 1 << 3,
#endif // _CCCL_HAS_BACKEND_TBB()
};

//! @brief Extracts the execution backend from the stored _Policy
template <uint32_t _Policy>
inline constexpr __execution_backend __policy_to_execution_backend =
  __execution_backend{(_Policy & uint32_t{0x0000FF00}) >> 8};

template <uint32_t _Policy, __execution_backend _Backend = __policy_to_execution_backend<_Policy>>
struct __execution_policy_base;

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_EXECUTION_POLICY_H
