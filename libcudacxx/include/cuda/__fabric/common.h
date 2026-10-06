//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_COMMON_H
#define _CUDA___FABRIC_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>
#  include <cuda/std/limits>

#  include <cuda/std/__cccl/prologue.h>

//! @brief Device-side fabric transport operations.
//!
//! Each try_* call only issues a request: the caller supplies proxy fences, submits the request,
//! arrives on the barrier with the documented transaction count, and waits for completion/status.
//! Local operands and the barrier must reside in CTA shared memory, and endpoints must remain
//! ready and bound until completion. Inputs must remain unchanged until read consumption or
//! completion; outputs must not be used until completion. Requests must complete before grid exit.
_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC
template <class _Tp>
_CCCL_DEVICE_API inline void
__check_transfer(::cuda::std::uint64_t __offset, const _Tp* __shared, ::cuda::std::size_t __bytes) noexcept
{
  _CCCL_ASSERT(__bytes != 0 && __bytes % 16 == 0, "fabric transfer size must be a nonzero multiple of 16");
  _CCCL_ASSERT(__bytes <= ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max(), "fabric size exceeds PTX operand");
  _CCCL_ASSERT(__offset % 16 == 0 && reinterpret_cast<::cuda::std::uintptr_t>(__shared) % 16 == 0,
               "fabric transfer operands must be 16B aligned");
  _CCCL_ASSERT(::__isShared(__shared), "fabric local operand must be CTA shared memory");
}
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_COMMON_H
