//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_RESTRICT_H
#define __CCCL_RESTRICT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(__cplusplus) && defined(_CCCL_COMPILER_MSVC) // vvv _CCCL_COMPILER_MSVC vvv

#  define _CCCL_RESTRICT __restrict

#elif defined(__cplusplus) // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv

#  define _CCCL_RESTRICT __restrict__

#elif !defined(__cplusplus) && __STDC_VERSION__ >= 199901L // ^^^ C++ ^^^ / vvv C99 vvv

#  define _CCCL_RESTRICT restrict

#else // ^^^ C99 ^^^ / vvv !C99 vvv

#  define _CCCL_RESTRICT

#endif // ^^^ !C99 ^^^

#endif // __CCCL_RESTRICT_H
