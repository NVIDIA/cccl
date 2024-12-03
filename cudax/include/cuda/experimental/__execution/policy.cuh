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

#include <cuda/experimental/__async/utility.cuh>

namespace cuda::experimental::execution
{

enum class execution_policy
{
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
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy =
  _Policy == execution_policy::parallel_host || _Policy == execution_policy::parallel_device
  || _Policy == execution_policy::parallel_unsequenced_host || _Policy == execution_policy::parallel_unsequenced_device;

template <execution_policy _Policy>
_CCCL_INLINE_VAR constexpr bool __is_unsequenced_execution_policy =
  _Policy == execution_policy::unsequenced_host || _Policy == execution_policy::unsequenced_device
  || _Policy == execution_policy::parallel_unsequenced_host || _Policy == execution_policy::parallel_unsequenced_device;

} // namespace cuda::experimental::execution

#endif //__CUDAX___EXECUTION_POLICY_CUH
