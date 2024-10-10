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

/// In device code, _CCCL_PTX_ARCH expands to the PTX version for which we are compiling.
/// In host code, _CCCL_PTX_ARCH's value is implementation defined.
#ifndef _CCCL_PTX_ARCH
#  if defined(_CCCL_COMPILER_NVHPC)
// __NVCOMPILER_CUDA_ARCH__ is the target PTX version, and is defined when compiling both host code and device code.
// Currently, only one PTX version can be targeted.
#    define _CCCL_PTX_ARCH __NVCOMPILER_CUDA_ARCH__
#  elif !defined(__CUDA_ARCH__)
#    define _CCCL_PTX_ARCH 0
#  else
#    define _CCCL_PTX_ARCH __CUDA_ARCH__
#  endif
#endif

#if defined(_CCCL_CUDA_COMPILER_NVCC) && _CCCL_PTX_ARCH >= 700 && !defined(_CCCL_CUDACC_BELOW_11_7)

#  define _CCCL_GRID_CONSTANT __grid_constant__

#else // ^^^ _CCCL_CUDA_COMPILER_NVCC ^^^ / vvv !_CCCL_CUDA_COMPILER_NVCC vvv

#  define _CCCL_GRID_CONSTANT

#endif // !_CCCL_CUDA_COMPILER_NVCC

#endif // __CCCL_GRID_CONSTANT_H
