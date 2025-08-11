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

#include <cuda/__memory/address_space.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API inline bool __is_valid_address_range(const void* __ptr, _CUDA_VSTD::size_t __n) noexcept
{
  if (__ptr == nullptr)
  {
    return false;
  }
  if (::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::shared))
  {
    if (_CUDA_VSTD::cmp_greater(__n, _CUDA_VSTD::numeric_limits<uint32_t>::max()))
    {
      return false;
    }
    auto __limit = size_t{_CUDA_VSTD::numeric_limits<_CUDA_VSTD::uint32_t>::max()} - __n;
    return reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) <= __limit;
  }
  auto __limit = size_t{_CUDA_VSTD::numeric_limits<_CUDA_VSTD::uintptr_t>::max()} - __n;
  return reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) <= __limit;
}

[[nodiscard]] _CCCL_API inline bool __is_valid_address(const void* __ptr) noexcept
{
  return __is_valid_address_range(__ptr, 0);
}

[[nodiscard]] _CCCL_API inline bool
__are_ptrs_overlapping(const void* __ptr_lhs, const void* __ptr_rhs, _CUDA_VSTD::size_t __n) noexcept
{
  auto __ptr1 = static_cast<const char*>(__ptr_lhs);
  auto __ptr2 = static_cast<const char*>(__ptr_rhs);
  return ((__ptr1 + __n) < __ptr2) || ((__ptr2 + __n) < __ptr1);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_VALID_ADDRESS
