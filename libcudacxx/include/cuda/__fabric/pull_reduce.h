//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_PULL_REDUCE_H
#define _CUDA___FABRIC_PULL_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__fabric/common.h>
#  include <cuda/__logical_endpoint/multicast.h>
#  include <cuda/__ptx/instructions/fabric_try_pullred.h>
#  include <cuda/barrier>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC
//! @brief Issue a warp-collective addition across the multicast source resources.
//! @note Completion contributes __bytes transactions to @p __barrier.
//! @pre All 32 warp lanes execute this call with identical arguments; no lane has exited.
//! @pre The issuing device belongs to the multicast endpoint, which supports pull-reduction.
//! @param[in] __src Ready, bound source endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[out] __shared Local CTA shared-memory destination, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_pull_reduce_add(
  ::cuda::multicast_logical_endpoint_ref __src,
  ::cuda::std::uint64_t __offset,
  _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_pullred(
    ::cuda::ptx::op_add,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    ::cuda::ptx::space_shared,
    __shared,
    __src.native_handle(),
    __offset,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue a warp-collective minimum across the multicast source resources.
//! @note Completion contributes __bytes transactions to @p __barrier.
//! @pre All 32 warp lanes execute this call with identical arguments; no lane has exited.
//! @pre The issuing device belongs to the multicast endpoint, which supports pull-reduction.
//! @param[in] __src Ready, bound source endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[out] __shared Local CTA shared-memory destination, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_pull_reduce_min(
  ::cuda::multicast_logical_endpoint_ref __src,
  ::cuda::std::uint64_t __offset,
  _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_pullred(
    ::cuda::ptx::op_min,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    ::cuda::ptx::space_shared,
    __shared,
    __src.native_handle(),
    __offset,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue a warp-collective maximum across the multicast source resources.
//! @note Completion contributes __bytes transactions to @p __barrier.
//! @pre All 32 warp lanes execute this call with identical arguments; no lane has exited.
//! @pre The issuing device belongs to the multicast endpoint, which supports pull-reduction.
//! @param[in] __src Ready, bound source endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[out] __shared Local CTA shared-memory destination, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_pull_reduce_max(
  ::cuda::multicast_logical_endpoint_ref __src,
  ::cuda::std::uint64_t __offset,
  _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_pullred(
    ::cuda::ptx::op_max,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    ::cuda::ptx::space_shared,
    __shared,
    __src.native_handle(),
    __offset,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_PULL_REDUCE_H
