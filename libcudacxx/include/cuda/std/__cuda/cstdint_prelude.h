// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_CSTDINT_PRELUDE_H
#define _LIBCUDACXX___CUDA_CSTDINT_PRELUDE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER_NVRTC
#  include <cstdint>
#else // ^^^ !_CCCL_COMPILER_NVRTC ^^^ / vvv _CCCL_COMPILER_NVRTC vvv
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed long long int64_t;
typedef unsigned long long uint64_t;

#  define _LIBCUDACXX_ADDITIONAL_INTS(N)  \
    typedef int##N##_t int_fast##N##_t;   \
    typedef uint##N##_t uint_fast##N##_t; \
    typedef int##N##_t int_least##N##_t;  \
    typedef uint##N##_t uint_least##N##_t

_LIBCUDACXX_ADDITIONAL_INTS(8);
_LIBCUDACXX_ADDITIONAL_INTS(16);
_LIBCUDACXX_ADDITIONAL_INTS(32);
_LIBCUDACXX_ADDITIONAL_INTS(64);
#  undef _LIBCUDACXX_ADDITIONAL_INTS

typedef int64_t intptr_t;
typedef uint64_t uintptr_t;
typedef int64_t intmax_t;
typedef uint64_t uintmax_t;

#  define INT8_MIN        SCHAR_MIN
#  define INT16_MIN       SHRT_MIN
#  define INT32_MIN       INT_MIN
#  define INT64_MIN       LLONG_MIN
#  define INT8_MAX        SCHAR_MAX
#  define INT16_MAX       SHRT_MAX
#  define INT32_MAX       INT_MAX
#  define INT64_MAX       LLONG_MAX
#  define UINT8_MAX       UCHAR_MAX
#  define UINT16_MAX      USHRT_MAX
#  define UINT32_MAX      UINT_MAX
#  define UINT64_MAX      ULLONG_MAX
#  define INT_FAST8_MIN   SCHAR_MIN
#  define INT_FAST16_MIN  SHRT_MIN
#  define INT_FAST32_MIN  INT_MIN
#  define INT_FAST64_MIN  LLONG_MIN
#  define INT_FAST8_MAX   SCHAR_MAX
#  define INT_FAST16_MAX  SHRT_MAX
#  define INT_FAST32_MAX  INT_MAX
#  define INT_FAST64_MAX  LLONG_MAX
#  define UINT_FAST8_MAX  UCHAR_MAX
#  define UINT_FAST16_MAX USHRT_MAX
#  define UINT_FAST32_MAX UINT_MAX
#  define UINT_FAST64_MAX ULLONG_MAX

#  define INT8_C(X)    ((int_least8_t) (X))
#  define INT16_C(X)   ((int_least16_t) (X))
#  define INT32_C(X)   ((int_least32_t) (X))
#  define INT64_C(X)   ((int_least64_t) (X))
#  define UINT8_C(X)   ((uint_least8_t) (X))
#  define UINT16_C(X)  ((uint_least16_t) (X))
#  define UINT32_C(X)  ((uint_least32_t) (X))
#  define UINT64_C(X)  ((uint_least64_t) (X))
#  define INTMAX_C(X)  ((intmax_t) (X))
#  define UINTMAX_C(X) ((uintmax_t) (X))
#endif // _CCCL_COMPILER_NVRTC

#endif // _LIBCUDACXX___CUDA_CSTDINT_PRELUDE_H
