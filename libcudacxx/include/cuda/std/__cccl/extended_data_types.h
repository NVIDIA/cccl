//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_EXTENDED_DATA_TYPES_H
#define __CCCL_EXTENDED_DATA_TYPES_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/diagnostic.h>
#include <cuda/std/__cccl/os.h>
#include <cuda/std/__cccl/preprocessor.h>

#if !defined(_CCCL_DISABLE_INT128)
#  if _CCCL_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__) && _CCCL_OS(LINUX)
#    define _CCCL_HAS_INT128() 1
#  elif defined(__SIZEOF_INT128__) && _CCCL_OS(LINUX) && _CCCL_CUDACC_AT_LEAST(11, 5)
#    define _CCCL_HAS_INT128() 1
#  else
#    define _CCCL_HAS_INT128() 0
#  endif
#else
#  define _CCCL_HAS_INT128() 0
#endif // !_CCCL_DISABLE_INT128

#if !defined(_CCCL_HAS_NVFP16)
#  if _CCCL_HAS_INCLUDE(<cuda_fp16.h>) && (_CCCL_HAS_CUDA_COMPILER || defined(LIBCUDACXX_ENABLE_HOST_NVFP16)) \
                        && !defined(CCCL_DISABLE_FP16_SUPPORT)
#    define _CCCL_HAS_NVFP16 1
#  endif
#endif // !_CCCL_HAS_NVFP16

#if !defined(_CCCL_HAS_NVBF16)
#  if _CCCL_HAS_INCLUDE(<cuda_bf16.h>) && defined(_CCCL_HAS_NVFP16) && !defined(CCCL_DISABLE_BF16_SUPPORT) \
                        && !defined(CUB_DISABLE_BF16_SUPPORT)
#    define _CCCL_HAS_NVBF16 1
#  endif
#endif // !_CCCL_HAS_NVBF16

#if !defined(_CCCL_DISABLE_NVFP8_SUPPORT)
#  if _CCCL_HAS_INCLUDE(<cuda_fp8.h>) && defined(_CCCL_HAS_NVFP16) && defined(_CCCL_HAS_NVBF16)
#    define _CCCL_HAS_NVFP8() 1
#  else
#    define _CCCL_HAS_NVFP8() 0
#  endif // _CCCL_HAS_INCLUDE(<cuda_fp8.h>) && defined(_CCCL_HAS_NVFP16) && defined(_CCCL_HAS_NVBF16)
#else
#  define _CCCL_HAS_NVFP8() 0
#endif // !defined(_CCCL_DISABLE_NVFP8_SUPPORT)

#if !defined(_CCCL_DISABLE_FLOAT128)
#  if _CCCL_COMPILER(NVRTC) && defined(__CUDACC_RTC_FLOAT128__) && _CCCL_OS(LINUX)
#    if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
#      define _CCCL_HAS_FLOAT128() 1
#    else
#      define _CCCL_HAS_FLOAT128() 0
#    endif
// NVC++ support float128 only in host code
#  elif (defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__)) && _CCCL_OS(LINUX) && !_CCCL_CUDA_COMPILER(NVHPC)
#    if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
#      define _CCCL_HAS_FLOAT128() 1
#    else
#      define _CCCL_HAS_FLOAT128() 0
#    endif
#  else
#    define _CCCL_HAS_FLOAT128() 0
#  endif
#endif // !defined(_CCCL_DISABLE_FLOAT128)

#endif // __CCCL_EXTENDED_DATA_TYPES_H
