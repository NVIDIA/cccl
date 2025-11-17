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

#include <cuda/__memcpy_async/group_traits.h>
#include <cuda/__ptx/instructions/elect_sync.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

//! Elects a single leader thread from a one dimensional thread block. For SM90+ will use ptx::elect_sync() etc.,
//! otherwise just selects the thread with ID 0. If the returned value is used as condition for an if statement, the
//! compiler will emit a uniform data path for the branch.
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __block_elect_one() noexcept
{
  _CCCL_ASSERT(blockDim.y == 1 && blockDim.z == 1, "The block must by one dimensional");

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (const auto tid             = threadIdx.x; //
     const auto warp_id         = tid / 32;
     const auto uniform_warp_id = ::__shfl_sync(~0, warp_id, 0); // broadcast from lane 0
     return uniform_warp_id == 0 && ::cuda::ptx::elect_sync(~0); // elect a leader thread among warp 0
     ),
    (return threadIdx.x == 0;));
}

template <typename _Group>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool __group_elect_one(const _Group& __g) noexcept
{
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                 if constexpr (is_thread_block_group_v<_Group>) {
                   // cooperative groups maps a multidimensional thread id into the thread rank the same way as warps do
                   const unsigned __tid             = __g.thread_rank();
                   const unsigned __warp_id         = __tid / 32;
                   const unsigned __uniform_warp_id = ::__shfl_sync(~0, __warp_id, 0); // broadcast from lane 0
                   return __uniform_warp_id == 0 && ::cuda::ptx::elect_sync(~0); // elect a leader thread among warp 0
                 } else if constexpr (is_warp_group_v<_Group>) { return ::cuda::ptx::elect_sync(~0); }));

  return __g.thread_rank() == 0;
}

#endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_DEVICE

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMCPY_ASYNC_ELECT_ONE_H_
