//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_UNREACHABLE_H
#define __CCCL_UNREACHABLE_H

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/visibility.h>

#if _CCCL_COMPILER(MSVC) && !defined(__CUDA_ARCH__)

#  define _CCCL_UNREACHABLE()             \
    {                                     \
      _CCCL_ASSERT(false, "unreachable"); \
      __assume(0);                        \
    }

#else

#  define _CCCL_UNREACHABLE()             \
    {                                     \
      _CCCL_ASSERT(false, "unreachable"); \
      __builtin_unreachable();            \
    }

#endif // _CCCL_COMPILER(MSVC) && !defined(__CUDA_ARCH__)

#endif // __CCCL_UNREACHABLE_H
