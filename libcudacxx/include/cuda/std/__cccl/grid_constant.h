//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_GRID_CONSTANT_H
#define __CCCL_GRID_CONSTANT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_CCCL_CUDA_COMPILER) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && !defined(_CCCL_CUDACC_BELOW_11_7)

#  define _CCCL_GRID_CONSTANT__ __grid_constant__

#else // ^^^ _CCCL_CUDA_COMPILER ^^^ / vvv !_CCCL_CUDA_COMPILER vvv

#  define _CCCL_GRID_CONSTANT__

#endif // !_CCCL_CUDA_COMPILER

#endif // __CCCL_GRID_CONSTANT_H
