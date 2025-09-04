//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_IS_VALID_ADDRESS
#define _CUDA___MEMORY_IS_VALID_ADDRESS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// including cuda/std/limits generates a circular dependency because:
//    numeric_limits -> bit_cast -> cstring -> check_address
// <cuda/std/__utility/cmp.h> also includes cuda/std/limits
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__memory/address_space.h>
#endif // _CCCL_CUDA_COMPILATION()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

[[nodiscard]] _CCCL_DEVICE_API inline bool
__is_smem_valid_address_range(const void* __ptr, ::cuda::std::size_t __n) noexcept
{
  if (::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::shared))
  {
    // clang-format off
    NV_IF_TARGET(NV_PROVIDES_SM_90, ( // smem can start at address 0x0 before sm_90
      if (__ptr == nullptr)
      {
        return false;
      })
    );
    // clang-format on
    if (__n > ::cuda::std::size_t{UINT32_MAX})
    {
      return false;
    }
    auto __limit = ::cuda::std::uintptr_t{UINT32_MAX} - static_cast<::cuda::std::uintptr_t>(__n);
    return reinterpret_cast<::cuda::std::uintptr_t>(__ptr) <= __limit;
  }
  return true;
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#endif // _CCCL_CUDA_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API inline bool __is_valid_address_range(const void* __ptr, ::cuda::std::size_t __n) noexcept
{
  // clang-format off
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (
    if (!::cuda::device::__is_smem_valid_address_range(__ptr, __n))
    {
      return false;
    };),
   (if (__ptr == nullptr)
    {
      return false;
    })
  );
  // clang-format on
  auto __limit = ::cuda::std::uintptr_t{UINTMAX_MAX} - static_cast<::cuda::std::uintptr_t>(__n);
  return reinterpret_cast<::cuda::std::uintptr_t>(__ptr) <= __limit;
}

[[nodiscard]] _CCCL_API inline bool __is_valid_address(const void* __ptr) noexcept
{
  return ::cuda::__is_valid_address_range(__ptr, 0);
}

[[nodiscard]] _CCCL_API inline bool
__are_ptrs_overlapping(const void* __ptr_lhs, const void* __ptr_rhs, ::cuda::std::size_t __n) noexcept
{
  auto __ptr1 = static_cast<const char*>(__ptr_lhs);
  auto __ptr2 = static_cast<const char*>(__ptr_rhs);
  return ((__ptr1 + __n) < __ptr2) || ((__ptr2 + __n) < __ptr1);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_VALID_ADDRESS
