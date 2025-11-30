//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_IS_ALIGNED_H
#define _CUDA___MEMORY_IS_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_is_aligned)
#  define _CCCL_BUILTIN_IS_ALIGNED(...) __builtin_is_aligned(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_is_aligned)

// nvcc doesn't support this builtin in device code, clang-cuda crashes
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_BUILTIN_IS_ALIGNED
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

[[nodiscard]] _CCCL_API inline bool is_aligned(const void* __ptr, ::cuda::std::size_t __alignment) noexcept
{
  _CCCL_ASSERT(::cuda::is_power_of_two(__alignment), "alignment must be a power of two");
#if defined(_CCCL_BUILTIN_IS_ALIGNED)
  return _CCCL_BUILTIN_IS_ALIGNED(__ptr, __alignment);
#else // ^^^ _CCCL_BUILTIN_IS_ALIGNED ^^^ / vvv !_CCCL_BUILTIN_IS_ALIGNED vvv
  return (reinterpret_cast<::cuda::std::uintptr_t>(__ptr) & (__alignment - 1)) == 0;
#endif // ^^^ !_CCCL_BUILTIN_IS_ALIGNED ^^^
}

[[nodiscard]] _CCCL_API inline bool is_aligned(const volatile void* __ptr, ::cuda::std::size_t __alignment) noexcept
{
  return ::cuda::is_aligned(const_cast<const void*>(__ptr), __alignment);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_IS_ALIGN_H
