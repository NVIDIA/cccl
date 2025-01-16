//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_128BIT_TYPES_H
#define __CCCL_128BIT_TYPES_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/os.h>

#ifndef _CCCL_DISABLE_INT128
#  if defined(__SIZEOF_INT128__) // defined(__SIZEOF_INT128__) vvvv
#    if _CCCL_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__) && !_CCCL_OS(WINDOWS)
#      define _CCCL_HAS_INT128() 1
#    elif (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)) && !_CCCL_OS(WINDOWS)
#      define _CCCL_HAS_INT128() 1
#    else
#      define _CCCL_HAS_INT128() 0
#    endif
#  else // defined(__SIZEOF_INT128__) ^^^^ / !defined(__SIZEOF_INT128__) vvvv
#    define _CCCL_HAS_INT128() 0
#  endif // !defined(__SIZEOF_INT128__) ^^^^
#else
#  define _CCCL_HAS_INT128() 0
#endif // #ifndef _CCCL_DISABLE_INT128

#ifndef _CCCL_DISABLE_FLOAT128
#  if (defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__)) && _CCCL_OS(LINUX) \
    && (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)) && !defined(__CUDA_ARCH__)
#    define _CCCL_HAS_FLOAT128() 1
#  else
#    define _CCCL_HAS_FLOAT128() 0
#  endif
#else
#  define _CCCL_HAS_INT128() 0
#endif // #ifndef _CCCL_DISABLE_FLOAT128

#endif // __CCCL_128BIT_TYPES_H
