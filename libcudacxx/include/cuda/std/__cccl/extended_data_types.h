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

#include <cuda/std/__cccl/architecture.h>
#include <cuda/std/__cccl/cuda_toolkit.h>
#include <cuda/std/__cccl/diagnostic.h>
#include <cuda/std/__cccl/os.h>
#include <cuda/std/__cccl/preprocessor.h>

#define _CCCL_HAS_INT128()      0
#define _CCCL_HAS_LONG_DOUBLE() 0
#define _CCCL_HAS_NVFP4()       0
#define _CCCL_HAS_NVFP4_E2M1()  _CCCL_HAS_NVFP4()
#define _CCCL_HAS_NVFP6()       0
#define _CCCL_HAS_NVFP6_E2M3()  _CCCL_HAS_NVFP6()
#define _CCCL_HAS_NVFP6_E3M2()  _CCCL_HAS_NVFP6()
#define _CCCL_HAS_NVFP8()       0
#define _CCCL_HAS_NVFP8_E4M3()  _CCCL_HAS_NVFP8()
#define _CCCL_HAS_NVFP8_E5M2()  _CCCL_HAS_NVFP8()
#define _CCCL_HAS_NVFP8_E8M0()  (_CCCL_HAS_NVFP8() && _CCCL_CUDACC_AT_LEAST(12, 8))
#define _CCCL_HAS_NVFP16()      0
#define _CCCL_HAS_NVBF16()      0
#define _CCCL_HAS_FLOAT128()    0

#define _CCCL_HAS_FLOAT128_LITERAL() _CCCL_HAS_FLOAT128()

#if !defined(CCCL_DISABLE_INT128_SUPPORT) && _CCCL_OS(LINUX) \
  && ((_CCCL_COMPILER(NVRTC) && defined(__CUDACC_RTC_INT128__)) || defined(__SIZEOF_INT128__))
#  undef _CCCL_HAS_INT128
#  define _CCCL_HAS_INT128() 1
#endif

// FIXME: Enable this for clang-cuda in a followup
#if !_CCCL_HAS_CUDA_COMPILER()
#  undef _CCCL_HAS_LONG_DOUBLE
#  define _CCCL_HAS_LONG_DOUBLE() 1
#endif // !_CCCL_HAS_CUDA_COMPILER()

#if _CCCL_HAS_INCLUDE(<cuda_fp16.h>) && (_CCCL_HAS_CTK() || defined(LIBCUDACXX_ENABLE_HOST_NVFP16)) \
                      && !defined(CCCL_DISABLE_FP16_SUPPORT)
#  undef _CCCL_HAS_NVFP16
#  define _CCCL_HAS_NVFP16() 1
#endif

#if _CCCL_HAS_INCLUDE(<cuda_bf16.h>) && _CCCL_HAS_NVFP16() && !defined(CCCL_DISABLE_BF16_SUPPORT)
#  undef _CCCL_HAS_NVBF16
#  define _CCCL_HAS_NVBF16() 1
#endif

#if _CCCL_HAS_INCLUDE(<cuda_fp8.h>) && _CCCL_HAS_NVFP16() && _CCCL_HAS_NVBF16() && !defined(CCCL_DISABLE_NVFP8_SUPPORT)
#  undef _CCCL_HAS_NVFP8
#  define _CCCL_HAS_NVFP8() 1
#endif

#if _CCCL_HAS_INCLUDE(<cuda_fp6.h>) && _CCCL_HAS_NVFP8() && !defined(CCCL_DISABLE_NVFP6_SUPPORT)
#  undef _CCCL_HAS_NVFP6
#  define _CCCL_HAS_NVFP6() 1
#endif

#if _CCCL_HAS_INCLUDE(<cuda_fp4.h>) && _CCCL_HAS_NVFP6() && !defined(CCCL_DISABLE_NVFP4_SUPPORT)
#  undef _CCCL_HAS_NVFP4
#  define _CCCL_HAS_NVFP4() 1
#endif

#if !defined(CCCL_DISABLE_FLOAT128_SUPPORT) && _CCCL_OS(LINUX) && !_CCCL_ARCH(ARM64)
#  if (defined(__CUDACC_RTC_FLOAT128__) || defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__)) /*HOST COMPILERS*/
#    if _CCCL_CUDA_COMPILER(NVHPC) \
      || ((_CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(CLANG)) && __CUDA_ARCH__ >= 1000) /*DEVICE CODE*/
#      undef _CCCL_HAS_FLOAT128
#      define _CCCL_HAS_FLOAT128() 1
#    endif // CUDA compiler
#  endif // Host compiler support
#endif // !CCCL_DISABLE_FLOAT128_SUPPORT && _CCCL_OS(LINUX)

// gcc does not allow to use 'operator""q' when __STRICT_ANSI__ is defined, it may be allowed by
// -fext-numeric-literals, but we have no way to detect it. However, from gcc 13, we can use 'operator""f128' and cast
// it to __float128.
#if _CCCL_COMPILER(GCC, >=, 13)
#  define _CCCL_FLOAT128_LITERAL(_X) __float128(_X##f128)
#elif !(_CCCL_COMPILER(GCC) && defined(__STRICT_ANSI__))
#  define _CCCL_FLOAT128_LITERAL(_X) __float128(_X##q)
#else // ^^^ has __float128 literal ^^^ // vvv no __float128 literal vvv
#  undef _CCCL_HAS_FLOAT128_LITERAL
#  define _CCCL_HAS_FLOAT128_LITERAL() 0
#endif // ^^^ no __float128 literal ^^^

#endif // __CCCL_EXTENDED_DATA_TYPES_H
