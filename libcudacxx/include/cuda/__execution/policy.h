//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EXECUTION_POLICY_H
#define _CUDA___EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__execution/policy.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

using __cub_parallel_unsequenced_policy =
  ::cuda::std::execution::__execution_policy_base<::cuda::std::execution::__with_backend<
    static_cast<uint32_t>(::cuda::std::execution::__execution_policy::__parallel_unsequenced),
    ::cuda::std::execution::__execution_backend::__cuda>()>;
_CCCL_GLOBAL_CONSTANT __cub_parallel_unsequenced_policy gpu{};

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA___EXECUTION_POLICY_H
