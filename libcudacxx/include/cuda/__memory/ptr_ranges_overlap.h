//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_PTR_RANGES_OVERLAP_H
#define _CUDA___MEMORY_PTR_RANGES_OVERLAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/ptr_in_range.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API constexpr bool ptr_ranges_overlap(
  const void* __ptr_lhs_start,
  const void* __ptr_lhs_end,
  const void* __ptr_rhs_start,
  const void* __ptr_rhs_end) noexcept
{
  return ::cuda::ptr_in_range(__ptr_lhs_start, __ptr_rhs_start, __ptr_rhs_end)
      || ::cuda::ptr_in_range(__ptr_rhs_start, __ptr_lhs_start, __ptr_lhs_end);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_PTR_RANGES_OVERLAP_H
