//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ALIGN_DOWN_H
#define _CUDA___MEMORY_ALIGN_DOWN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
[[nodiscard]] _CCCL_API _CUDA_VSTD::enable_if_t<_CUDA_VSTD::is_pointer_v<_Tp>, _Tp>
align_up(_Tp __ptr, size_t __alignment) noexcept
{
  _CCCL_ASSERT(::cuda::is_power_of_two(__alignment), "alignment must be a power of two");
  _CCCL_ASSERT(reinterpret_cast<uintptr_t>(__ptr) % alignof(_Tp) == 0, "ptr is not aligned");
  if (alignof(_Tp) >= __alignment)
  {
    return __ptr;
  }
  auto __ret = reinterpret_cast<uintptr_t>(__ptr) & ~static_cast<uintptr_t>(__alignment - 1);
  return _CCCL_BUILTIN_ASSUME_ALIGNED(reinterpret_cast<_Tp>(__ret), __alignment);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_ALIGN_DOWN_H
