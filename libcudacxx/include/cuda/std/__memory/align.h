// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ALIGN_H
#define _LIBCUDACXX___MEMORY_ALIGN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4146) // unary minus operator applied to unsigned type, result still unsigned

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI void* align(size_t __alignment, size_t __size, void*& __ptr, size_t& __space)
{
  if (__space < __size)
  {
    return nullptr;
  }

  const auto __intptr  = reinterpret_cast<uintptr_t>(__ptr);
  const auto __aligned = (__intptr - 1u + __alignment) & -__alignment;
  const auto __diff    = __aligned - __intptr;
  if (__diff > (__space - __size))
  {
    return nullptr;
  }
  __space -= __diff;
  return __ptr = reinterpret_cast<void*>(__aligned);
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_DIAG_POP

#endif // _LIBCUDACXX___MEMORY_ALIGN_H
