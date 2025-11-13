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
#    include <cuda/std/__type_traits/conditional.h>
#    include <cuda/std/cstdint>

#    include <nv/target>

#    include <cuda/std/__cccl/prologue.h>

// forward declare cooperative groups types. we cannot include <cooperative_groups.h> since it does not work with NVHPC
namespace cooperative_groups
{
namespace __v1
{
class thread_block;

template <unsigned int Size, typename ParentT>
class thread_block_tile;
} // namespace __v1
using namespace __v1;
} // namespace cooperative_groups

_CCCL_BEGIN_NAMESPACE_CUDA

//! Trait to detect whether a group represents a CUDA thread block, for example: ``cooperative_groups::thread_block``.
template <typename _Group>
_CCCL_GLOBAL_CONSTANT bool is_thread_block_group_v = false;

template <>
_CCCL_GLOBAL_CONSTANT bool is_thread_block_group_v<cooperative_groups::thread_block> = true;

//! Trait to detect whether a group represents a CUDA warp, for example:
//! ``cooperative_groups::thread_block_tile<32, ...>``.
template <typename _Group>
_CCCL_GLOBAL_CONSTANT bool is_warp_group_v = false;

template <typename _Parent>
_CCCL_GLOBAL_CONSTANT bool is_warp_group_v<cooperative_groups::thread_block_tile<32, _Parent>> = true;

template <typename _Group>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE bool __elect_from_group(const _Group& __g) noexcept
{
  if constexpr (is_thread_block_group_v<_Group>)
  {
    // cooperative groups maps a multidimensional thread id into the thread rank the same way as warps do
    const unsigned int __tid             = __g.thread_rank();
    const unsigned int __warp_id         = __tid / 32;
    const unsigned int __uniform_warp_id = __shfl_sync(0xFFFFFFFF, __warp_id, 0); // broadcast from lane 0
    return __uniform_warp_id == 0 && ::cuda::ptx::elect_sync(0xFFFFFFFF); // elect a leader thread among warp 0
  }
  else if constexpr (is_warp_group_v<_Group>)
  {
    return ::cuda::ptx::elect_sync(0xFFFFFFFF); // elect a leader thread among warp 0
  }
  else
  {
    return __g.thread_rank() == 0;
  }
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
        ::cuda::std::conditional_t<__cccl_ptx_isa >= 860, ::cuda::ptx::space_shared_t, ::cuda::ptx::space_cluster_t>{},
        ::cuda::ptx::space_global,
        __dest,
        __src,
        __size,
        __bar_handle);
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
