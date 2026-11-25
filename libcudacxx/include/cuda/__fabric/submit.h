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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
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
//!
//! @note Does not arrive on the barrier or wait for completion.
_CCCL_DEVICE_API inline void submit() noexcept
{
  ::cuda::ptx::fabric_submit();
}
//! @brief Submit only this thread's previously issued get and pull-reduction requests.
//!
//! @note Does not arrive on the barrier or wait for completion.
_CCCL_DEVICE_API inline void submit_restrict_fetching() noexcept
{
  ::cuda::ptx::fabric_submit_op_restrict_fetching();
}
#  if __cccl_ptx_isa >= 950
//! @brief Submit this thread's previously issued fabric requests and commit a fabric submit group.
//!
//! Compared to plain @c submit(), this creates a fabric submit group that can be waited on with @c wait_reads().
//!
//! @note Requires CUDA 13.5 and SM100 or later. Previously submitted requests are unaffected.
//! Does not arrive on the barrier or wait for completion.
_CCCL_DEVICE_API inline void submit_commit_group() noexcept
{
  asm volatile("fabric.submit.commit_group;" : : : "memory");
}

#  endif // __cccl_ptx_isa >= 950

//! @brief Wait for local shared-memory reads, allowing up to @p _PendingGroups recent submit groups to remain pending.
//!
//! @tparam _PendingGroups Number of recent fabric submit groups allowed to remain pending; defaults to zero,
//! which waits for all groups.
//! @note Before CUDA 13.5, submit groups were unavailable: only zero is supported,
//! and this waits for local shared-memory reads of requests submitted with @c submit().
//! @note CUDA 13.5 introduced @c submit_commit_group(). With CUDA 13.5 and later, this waits
//! only on groups created by that function; requests submitted with @c submit() are not tracked.
//! Before arriving on a barrier, wait for the submit groups containing the requests tracked by that barrier.
//! @note Requires SM100 or later. This permits input reuse; it does not establish remote completion or inspect status.
template <unsigned int _PendingGroups = 0>
_CCCL_DEVICE_API void wait_reads() noexcept
{
#  if __cccl_ptx_isa >= 950
  asm volatile("fabric.wait.sync_restrict::read %0;" : : "n"(_PendingGroups) : "memory");
#  else
  static_assert(_PendingGroups == 0, "nonzero pending fabric submit groups require CUDA 13.5 or later");
  ::cuda::ptx::fabric_wait();
#  endif // __cccl_ptx_isa >= 950
}
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_SUBMIT_H
