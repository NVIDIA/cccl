// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_ELECT_ONE_H_
#define _CUDA___MEMCPY_ASYNC_ELECT_ONE_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  if __cccl_ptx_isa >= 800

#    include <cuda/__ptx/instructions/elect_sync.h>

#    include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

extern "C" _CCCL_DEVICE void __cuda_elect_sync_is_not_supported_before_SM_90__();

//! Elects a single leader thread from a one dimensional thread block
_CCCL_DEVICE_API _CCCL_FORCEINLINE bool __elect_one()
{
  _CCCL_ASSERT(blockDim.y == 1 && blockDim.z == 1, "The block must by one dimensional");

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (const auto tid             = threadIdx.x; //
     const auto warp_id         = tid / 32;
     const auto uniform_warp_id = __shfl_sync(~0, warp_id, 0); // broadcast from lane 0
     return uniform_warp_id == 0 && cuda::ptx::elect_sync(~0); // elect a leader thread among warp 0
     ),
    (::cuda::device::__cuda_elect_sync_is_not_supported_before_SM_90__(); _CCCL_UNREACHABLE();));
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___MEMCPY_ASYNC_ELECT_ONE_H_
