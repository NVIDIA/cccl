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

#ifndef _LIBCUDACXX___MEMORY_IS_SUFFICIENTLY_ALIGNED_H
#define _LIBCUDACXX___MEMORY_IS_SUFFICIENTLY_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/cstddef> // size_t
#include <cuda/std/cstdint> // uintptr_t

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _ByteAlignment, class _ElementType>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool is_sufficiently_aligned(_ElementType* __ptr) noexcept
{
  return _CUDA_VSTD::bit_cast<uintptr_t>(__ptr) % _ByteAlignment == 0;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_IS_SUFFICIENTLY_ALIGNED_H
