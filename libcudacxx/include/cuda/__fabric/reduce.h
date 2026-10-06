//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_REDUCE_H
#define _CUDA___FABRIC_REDUCE_H

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
#  include <cuda/__ptx/instructions/fabric_try_red.h>
#  include <cuda/barrier>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC
//! @brief Issue an element-wise addition into the destination endpoint.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_add(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red(
    ::cuda::ptx::op_add,
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue an element-wise addition with a remote byte-completion counter.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The endpoint supports counted completion.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_counted_add(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_counted(
    ::cuda::ptx::op_add,
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

//! @brief Issue an element-wise minimum into the destination endpoint.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_min(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red(
    ::cuda::ptx::op_min,
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue an element-wise minimum with a remote byte-completion counter.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The endpoint supports counted completion.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_counted_min(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_counted(
    ::cuda::ptx::op_min,
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

//! @brief Issue an element-wise maximum into the destination endpoint.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_max(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red(
    ::cuda::ptx::op_max,
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue an element-wise maximum with a remote byte-completion counter.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The endpoint supports counted completion.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_counted_max(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_counted(
    ::cuda::ptx::op_max,
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

//! @brief Issue an element-wise addition into the destination endpoint.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The issuing device belongs to the multicast endpoint.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_add(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_multimem(
    ::cuda::ptx::op_add,
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue an element-wise addition with a remote byte-completion counter.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The endpoint supports counted completion.
//! @pre The issuing device belongs to the multicast endpoint.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_counted_add(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_multimem_counted(
    ::cuda::ptx::op_add,
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

//! @brief Issue an element-wise minimum into the destination endpoint.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The issuing device belongs to the multicast endpoint.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_min(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_multimem(
    ::cuda::ptx::op_min,
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue an element-wise minimum with a remote byte-completion counter.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The endpoint supports counted completion.
//! @pre The issuing device belongs to the multicast endpoint.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_counted_min(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_multimem_counted(
    ::cuda::ptx::op_min,
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

//! @brief Issue an element-wise maximum into the destination endpoint.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The issuing device belongs to the multicast endpoint.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_max(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_multimem(
    ::cuda::ptx::op_max,
    ::cuda::ptx::space_shared,
    ::cuda::ptx::sem_relaxed,
    ::cuda::ptx::scope_sys,
    __dst.native_handle(),
    __offset,
    __shared,
    static_cast<::cuda::std::uint32_t>(__bytes),
    ::cuda::device::barrier_native_handle(__barrier));
}

//! @brief Issue an element-wise maximum with a remote byte-completion counter.
//! @note Completion contributes __bytes / 16 transactions to @p __barrier.
//! @pre The endpoint supports counted completion.
//! @pre The issuing device belongs to the multicast endpoint.
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, aligned to 16 bytes.
//! @param[in] __counter_offset Endpoint byte offset of the 8-byte completion counter, aligned to 256 bytes.
//! @param[in] __shared Local CTA shared-memory source, aligned to 16 bytes.
//! @param[in] __bytes Nonzero byte count, a multiple of 16 that fits in uint32_t.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API inline void try_reduce_counted_max(
  ::cuda::multicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  ::cuda::std::uint64_t __counter_offset,
  const _Tp* __shared,
  ::cuda::std::size_t __bytes,
  ::cuda::shared_barrier& __barrier) noexcept
{
  ::cuda::fabric::__check_transfer(__offset, __shared, __bytes);
  ::cuda::ptx::fabric_try_red_multimem_counted(
    ::cuda::ptx::op_max,
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
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_REDUCE_H
