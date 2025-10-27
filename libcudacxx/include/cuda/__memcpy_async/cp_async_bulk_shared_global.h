// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_CP_ASYNC_BULK_SHARED_GLOBAL_H_
#define _CUDA___MEMCPY_ASYNC_CP_ASYNC_BULK_SHARED_GLOBAL_H_

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

#    include <cuda/__ptx/instructions/cp_async_bulk.h>
#    include <cuda/__ptx/instructions/elect_sync.h>
#    include <cuda/__ptx/instructions/mbarrier_expect_tx.h>
#    include <cuda/__ptx/ptx_dot_variants.h>
#    include <cuda/__ptx/ptx_helper_functions.h>
#    include <cuda/std/cstdint>

#    include <nv/target>

#    include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Group>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool __elect_from_group(const _Group& __g) noexcept
{
  // cooperative groups maps a multidimensional thread id into the thread rank the same way as warps do
  const unsigned int tid             = __g.thread_rank();
  const unsigned int warp_id         = tid / 32;
  const unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // broadcast from lane 0
  return uniform_warp_id == 0 && cuda::ptx::elect_sync(0xFFFFFFFF); // elect a leader thread among warp 0
}

extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_shared_global_is_not_supported_before_SM_90__();
template <typename _Group>
inline _CCCL_DEVICE void __cp_async_bulk_shared_global_and_expect_tx(
  const _Group& __g, char* __dest, const char* __src, ::cuda::std::size_t __size, ::cuda::std::uint64_t* __bar_handle)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (if (__elect_from_group(__g)) {
      ::cuda::ptx::cp_async_bulk(
        ::cuda::ptx::space_cluster, ::cuda::ptx::space_global, __dest, __src, __size, __bar_handle);
      ::cuda::ptx::mbarrier_expect_tx(
        ::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta, ::cuda::ptx::space_shared, __bar_handle, __size);
    }),
    (::cuda::__cuda_ptx_cp_async_bulk_shared_global_is_not_supported_before_SM_90__();));
}

_CCCL_END_NAMESPACE_CUDA

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___MEMCPY_ASYNC_CP_ASYNC_BULK_SHARED_GLOBAL_H_
