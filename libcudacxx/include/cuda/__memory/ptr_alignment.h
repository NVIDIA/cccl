//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_PTR_ALIGNMENT_H
#define _CUDA___MEMORY_PTR_ALIGNMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Get the alignment of a pointer, namely the largest power of two that divides the pointer address.
//!
//! @param  __ptr           the input pointer.
//! @param  __max_alignment the maximum alignment to consider.
//! @return The alignment of the pointer as a `size_t` value (always a power of two).
//! @pre    __ptr is not null.
//! @pre    __max_alignment is 0 or a power of two.
[[nodiscard]] _CCCL_API inline ::cuda::std::size_t
ptr_alignment(const void* __ptr, ::cuda::std::size_t __max_alignment = 0) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "ptr_alignment requires a non-null pointer");
  _CCCL_ASSERT(__max_alignment == 0 || ::cuda::is_power_of_two(__max_alignment),
               "max_alignment must be a power of two");
  const auto __addr = reinterpret_cast<::cuda::std::uintptr_t>(__ptr) | __max_alignment;
  return static_cast<::cuda::std::size_t>(__addr & (~__addr + 1));
}

[[nodiscard]] _CCCL_API inline ::cuda::std::size_t ptr_alignment(const volatile void* __ptr) noexcept
{
  return ::cuda::ptr_alignment(const_cast<const void*>(__ptr));
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_PTR_ALIGNMENT_H
