//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_SUBMIT_H
#define _CUDA___FABRIC_SUBMIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__ptx/instructions/fabric_submit.h>
#  include <cuda/__ptx/instructions/fabric_wait.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC
//! @brief Submit this thread's previously issued fabric requests.
//! @note Does not arrive on the barrier or wait for completion.
_CCCL_DEVICE_API inline void submit() noexcept
{
  ::cuda::ptx::fabric_submit();
}
//! @brief Submit only this thread's previously issued get and pull-reduction requests.
//! @note Does not arrive on the barrier or wait for completion.
_CCCL_DEVICE_API inline void submit_restrict_fetching() noexcept
{
  ::cuda::ptx::fabric_submit_op_restrict_fetching();
}
//! @brief Wait until submitted fabric requests have consumed their local shared-memory inputs.
//! @note This permits input reuse; it does not establish remote completion or inspect status.
_CCCL_DEVICE_API inline void wait_reads() noexcept
{
  ::cuda::ptx::fabric_wait();
}
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_SUBMIT_H
