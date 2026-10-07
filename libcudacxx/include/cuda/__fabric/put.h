//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_PUT_H
#define _CUDA___FABRIC_PUT_H

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
#  include <cuda/__logical_endpoint/unicast.h>
#  include <cuda/__ptx/instructions/fabric_try_put.h>
#  include <cuda/barrier>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC

//! @brief Issue a put from shared memory to the destination endpoint.
//!
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void
try_put(::cuda::unicast_logical_endpoint_ref __dst,
        ::cuda::std::uint64_t __offset,
        const _Tp* __shared,
        ::cuda::std::size_t __bytes,
        ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_put(
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue a put with a remote byte-completion counter.
//!
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//!
//! @pre The endpoint supports counted completion.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_put_counted(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_put_counted(
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __counter_offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue a put with a per-byte mask applied to each 16-byte chunk.
//!
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in] __mask Bit mask selecting the bytes written within each 16-byte chunk.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_put_masked(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::std::uint16_t __mask,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_put_cp_mask(
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier),
    __mask);
}

//! @brief Issue a put from shared memory to the destination endpoint.
//!
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//!
//! @pre The issuing device belongs to the multicast endpoint.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void
try_put(::cuda::multicast_logical_endpoint_ref __dst,
        ::cuda::std::uint64_t __offset,
        const _Tp* __shared,
        ::cuda::std::size_t __bytes,
        ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_put_multimem(
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue a put with a remote byte-completion counter.
//!
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//!
//! @pre The endpoint supports counted completion.
//! @pre The issuing device belongs to the multicast endpoint.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_put_counted(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_put_multimem_counted(
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __counter_offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue a put with a per-byte mask applied to each 16-byte chunk.
//!
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//!
//! @pre The issuing device belongs to the multicast endpoint.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in] __mask Bit mask selecting the bytes written within each 16-byte chunk.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_put_masked(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::std::uint16_t __mask,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_put_multimem_cp_mask(
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier),
    __mask);
}
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_PUT_H
