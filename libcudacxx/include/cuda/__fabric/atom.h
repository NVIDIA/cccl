//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_ATOM_H
#define _CUDA___FABRIC_ATOM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__fabric/atomic_block.h>
#  include <cuda/__logical_endpoint/unicast.h>
#  include <cuda/__memory/address_space.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#  include <cuda/barrier>
#  include <cuda/std/__floating_point/cuda_fp_types.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_FABRIC

//! @brief Issue an atomic addition and write the previous value to shared memory.
//!
//! @tparam _Tp cuda::std::uint32_t, cuda::std::uint64_t, float, double, __half2, or __nv_bfloat162.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_fetch_add(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint32_t>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "add.u32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint64_t>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "add.u64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, float>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "add.f32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, double>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "add.f64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  if _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__half2>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "add.f16x2 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat162>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "add.bf16x2 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  endif // _CCCL_HAS_NVBF16()
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic minimum and write the previous value to shared memory.
//!
//! @tparam _Tp cuda::std::uint32_t, cuda::std::uint64_t, __half2, or __nv_bfloat162.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_fetch_min(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint32_t>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "min.u32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint64_t>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "min.u64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  if _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__half2>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "min.f16x2 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat162>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "min.bf16x2 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  endif // _CCCL_HAS_NVBF16()
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic maximum and write the previous value to shared memory.
//!
//! @tparam _Tp cuda::std::uint32_t, cuda::std::uint64_t, __half2, or __nv_bfloat162.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_fetch_max(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint32_t>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "max.u32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint64_t>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "max.u64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  if _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__half2>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "max.f16x2 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat162>)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "max.bf16x2 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
#  endif // _CCCL_HAS_NVBF16()
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic bitwise AND and write the previous value to shared memory.
//!
//! @tparam _Tp A 4- or 8-byte type supported by atomic_block.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_fetch_and(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (sizeof(_Tp) == 4)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "and.b32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 8)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "and.b64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic bitwise OR and write the previous value to shared memory.
//!
//! @tparam _Tp A 4- or 8-byte type supported by atomic_block.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_fetch_or(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (sizeof(_Tp) == 4)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or."
                 "b32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 8)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.or."
                 "b64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic bitwise XOR and write the previous value to shared memory.
//!
//! @tparam _Tp A 4- or 8-byte type supported by atomic_block.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_fetch_xor(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (sizeof(_Tp) == 4)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "xor.b32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 8)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "xor.b64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic exchange and write the previous value to shared memory.
//!
//! @tparam _Tp A 4-, 8-, or 16-byte type supported by atomic_block.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory input block.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_exchange(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const atomic_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (sizeof(_Tp) == 4)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "exch.b32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 8)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "exch.b64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 16)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "exch.b128 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}

//! @brief Issue an atomic compare-exchange and write the previous value to shared memory.
//!
//! @tparam _Tp A 4-, 8-, or 16-byte type supported by atomic_block.
//!
//! @note Completion contributes one transaction to @p __barrier.
//!
//! @param[in] __dst Ready, bound destination endpoint.
//! @param[in] __offset Endpoint byte offset, a multiple of sizeof(_Tp).
//! @param[out] __old_shared Local CTA shared-memory result block.
//! @param[in] __operand_shared Local CTA shared-memory compare/desired blocks, aligned to 32 bytes.
//! @param[in,out] __barrier Status-reporting barrier in local CTA shared memory.
template <class _Tp>
_CCCL_DEVICE_API void try_compare_exchange(
  ::cuda::unicast_logical_endpoint_ref __dst,
  ::cuda::std::uint64_t __offset,
  atomic_block<_Tp>* __old_shared,
  const compare_exchange_block<_Tp>* __operand_shared,
  ::cuda::shared_barrier& __barrier) noexcept
{
  _CCCL_ASSERT(__offset % sizeof(_Tp) == 0, "invalid atomic endpoint offset");
  _CCCL_ASSERT(::cuda::device::is_address_from(__old_shared, ::cuda::device::address_space::shared),
               "atomic result block must be in CTA shared memory");
  _CCCL_ASSERT(::cuda::device::is_address_from(__operand_shared, ::cuda::device::address_space::shared),
               "atomic input block must be in CTA shared memory");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__old_shared) % alignof(atomic_block<_Tp>) == 0,
               "invalid atomic result block alignment");
  _CCCL_ASSERT(reinterpret_cast<::cuda::std::uintptr_t>(__operand_shared) % alignof(compare_exchange_block<_Tp>) == 0,
               "invalid atomic input block alignment");
  const auto __old     = ::cuda::ptx::__as_ptr_smem(__old_shared);
  const auto __operand = ::cuda::ptx::__as_ptr_smem(__operand_shared);
  const auto __bar     = ::cuda::ptx::__as_ptr_smem(::cuda::device::barrier_native_handle(__barrier));
  if constexpr (sizeof(_Tp) == 4)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "cas.b32 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 8)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "cas.b64 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else if constexpr (sizeof(_Tp) == 16)
  {
    asm volatile("fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys."
                 "cas.b128 [%0, %1], [%2], [%3], [%4];"
                 :
                 : "r"(__dst.native_handle()), "l"(__offset), "r"(__old), "r"(__operand), "r"(__bar)
                 : "memory");
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "unsupported fabric atomic type");
  }
}
_CCCL_END_NAMESPACE_CUDA_FABRIC

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_ATOM_H
