// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_IS_SUFFICIENTLY_ALIGNED_H
#define _CUDA_STD___MEMORY_IS_SUFFICIENTLY_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/cstddef> // size_t
#include <cuda/std/cstdint> // uintptr_t

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _ByteAlignment, class _ElementType>
[[nodiscard]] _CCCL_API inline bool is_sufficiently_aligned(_ElementType* __ptr) noexcept
{
  using ::cuda::std::uintptr_t;
  static_assert(::cuda::is_power_of_two(_ByteAlignment), "alignment must be a power of two");
  if constexpr (!::cuda::std::is_void_v<_ElementType>)
  {
    static_assert(_ByteAlignment % alignof(_ElementType) == 0,
                  "the alignment must be a multiple of the element alignment");
  }
  return (reinterpret_cast<uintptr_t>(__ptr) % _ByteAlignment) == 0;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_IS_SUFFICIENTLY_ALIGNED_H
