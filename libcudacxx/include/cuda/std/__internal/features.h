//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTERNAL_FEATURES_H
#define _LIBCUDACXX___INTERNAL_FEATURES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// If NVCC is not being used <complex> can safely use `long double` without warnings
#if !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_HAS_COMPLEX_LONG_DOUBLE
#endif // !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)

#ifndef _LIBCUDACXX_HAS_EXTERNAL_ATOMIC_IMP
#  define _LIBCUDACXX_HAS_EXTERNAL_ATOMIC_IMP
#endif // _LIBCUDACXX_HAS_EXTERNAL_ATOMIC_IMP

#ifndef _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
#  if _CCCL_HAS_CUDA_COMPILER || __cpp_aligned_new < 201606
#    define _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION
#  endif // _CCCL_HAS_CUDA_COMPILER || __cpp_aligned_new < 201606
#endif // _LIBCUDACXX_HAS_NO_ALIGNED_ALLOCATION

#ifndef _LIBCUDACXX_HAS_NO_CHAR8_T
#  if _CCCL_STD_VER <= 2017 || !defined(__cpp_char8_t)
#    define _LIBCUDACXX_HAS_NO_CHAR8_T
#  endif // _CCCL_STD_VER <= 2017 || !defined(__cpp_char8_t)
#endif // _LIBCUDACXX_HAS_NO_CHAR8_T

// We need `is_constant_evaluated` for clang and gcc. MSVC also needs extensive rework
#if !defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#  define _LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS
#elif _CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS
#elif _CCCL_COMPILER(MSVC)
#  define _LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS
#elif _CCCL_CUDACC_BELOW(11, 8)
#  define _LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS
#elif _CCCL_CUDA_COMPILER(CLANG)
#  define _LIBCUDACXX_HAS_NO_CONSTEXPR_COMPLEX_OPERATIONS
#endif // _CCCL_CUDA_COMPILER(CLANG)

#ifndef _LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS
#  define _LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS
#endif // _LIBCUDACXX_HAS_NO_CXX20_CHRONO_LITERALS

#ifndef _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#  define _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#endif // _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES

#ifndef _LIBCUDACXX_HAS_NO_INT128
#  if _CCCL_COMPILER(MSVC) || (_CCCL_COMPILER(NVRTC) && !defined(__CUDACC_RTC_INT128__)) \
    || _CCCL_CUDA_COMPILER(NVCC, <, 11, 5) || !defined(__SIZEOF_INT128__)
#    define _LIBCUDACXX_HAS_NO_INT128
#  endif
#endif // !_LIBCUDACXX_HAS_NO_INT128

#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
#  if _CCCL_HAS_CUDA_COMPILER
#    define _LIBCUDACXX_HAS_NO_LONG_DOUBLE
#  endif // _CCCL_HAS_CUDA_COMPILER
#endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE

#ifndef _LIBCUDACXX_HAS_NO_MONOTONIC_CLOCK
#  define _LIBCUDACXX_HAS_NO_MONOTONIC_CLOCK
#endif // _LIBCUDACXX_HAS_NO_MONOTONIC_CLOCK

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#  define _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
#endif // _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR

#ifndef _LIBCUDACXX_HAS_NO_WCHAR_H
#  define _LIBCUDACXX_HAS_NO_WCHAR_H
#endif // _LIBCUDACXX_HAS_NO_WCHAR_H

// libcu++ requires host device support for its tests. Until then restrict usage to at least 12.2
#ifndef _LIBCUDACXX_HAS_NVFP16
#  if defined(_CCCL_HAS_NVFP16) && _CCCL_CUDACC_AT_LEAST(12, 2) \
    && (_CCCL_HAS_CUDA_COMPILER || defined(LIBCUDACXX_ENABLE_HOST_NVFP16))
#    define _LIBCUDACXX_HAS_NVFP16
#  endif // _CCCL_HAS_NVFP16 && _CCCL_CUDACC_AT_LEAST(12, 2)
#endif // !_LIBCUDACXX_HAS_NVFP16

// libcu++ requires host device support for its tests. Until then restrict usage to at least 12.2
#ifndef _LIBCUDACXX_HAS_NVBF16
#  if defined(_CCCL_HAS_NVBF16) && _CCCL_CUDACC_AT_LEAST(12, 2)
#    define _LIBCUDACXX_HAS_NVBF16
#  endif // _CCCL_HAS_NVBF16 && _CCCL_CUDACC_AT_LEAST(12, 2)
#endif // !_LIBCUDACXX_HAS_NVBF16

// NVCC does not have a way of silencing non '_' prefixed UDLs
#if !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_HAS_STL_LITERALS
#endif // !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)

#endif // _LIBCUDACXX___INTERNAL_FEATURES_H
