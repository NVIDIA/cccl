//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_SYSTEM_HEADER_H
#define __CCCL_SYSTEM_HEADER_H

#include "../__cccl/compiler.h"

// Enforce that cccl headers are treated as system headers
#if defined(_CCCL_COMPILER_GCC) || defined(_CCCL_COMPILER_NVHPC) || defined(_CCCL_COMPILER_ICC) \
  || defined(_CCCL_COMPILER_ICC_LLVM)
#  define _CCCL_FORCE_SYSTEM_HEADER_GCC
#elif defined(_CCCL_COMPILER_CLANG)
#  define _CCCL_FORCE_SYSTEM_HEADER_CLANG
#elif defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_FORCE_SYSTEM_HEADER_MSVC
#endif // other compilers

// Potentially enable that cccl headers are treated as system headers
#if !defined(_CCCL_NO_SYSTEM_HEADER)                                                     \
  && !(defined(_CCCL_COMPILER_MSVC) && defined(_LIBCUDACXX_DISABLE_PRAGMA_MSVC_WARNING)) \
  && !defined(_CCCL_COMPILER_NVRTC) && !defined(_LIBCUDACXX_DISABLE_PRAGMA_GCC_SYSTEM_HEADER)
#  if defined(_CCCL_COMPILER_GCC) || defined(_CCCL_COMPILER_NVHPC) || defined(_CCCL_COMPILER_ICC) \
    || defined(_CCCL_COMPILER_ICC_LLVM)
#    define _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  elif defined(_CCCL_COMPILER_CLANG)
#    define _CCCL_IMPLICIT_SYSTEM_HEADER_CLANG
#  elif defined(_CCCL_COMPILER_MSVC)
#    define _CCCL_IMPLICIT_SYSTEM_HEADER_MSVC
#  endif // other compilers
#endif // Use system header

#endif // __CCCL_SYSTEM_HEADER_H
