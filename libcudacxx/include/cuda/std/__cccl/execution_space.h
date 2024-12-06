//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_EXECUTION_SPACE_H
#define __CCCL_EXECUTION_SPACE_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// We need to ensure that we not only compile with a cuda compiler but also compile cuda source files
#if _CCCL_HAS_CUDA_COMPILER && (defined(__CUDACC__) || defined(_NVHPC_CUDA))
#  define _CCCL_HOST        __host__
#  define _CCCL_DEVICE      __device__
#  define _CCCL_HOST_DEVICE __host__ __device__
#else // ^^^ _CCCL_CUDA_COMPILATION ^^^ / vvv !_CCCL_CUDA_COMPILATION vvv
#  define _CCCL_HOST
#  define _CCCL_DEVICE
#  define _CCCL_HOST_DEVICE
#endif // !_CCCL_CUDA_COMPILATION

/// In device code, _CCCL_PTX_ARCH expands to the PTX version for which we are compiling.
/// In host code, _CCCL_PTX_ARCH's value is implementation defined.
#if !defined(__CUDA_ARCH__)
#  define _CCCL_PTX_ARCH 0
#else
#  define _CCCL_PTX_ARCH __CUDA_ARCH__
#endif

// Compile with NVCC compiler and only device code, Volta+  GPUs
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_PTX_ARCH >= 700 && _CCCL_CUDACC_AT_LEAST(11, 7)
#  define _CCCL_GRID_CONSTANT __grid_constant__
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC) vvv
#  define _CCCL_GRID_CONSTANT
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_PTX_ARCH >= 700 && _CCCL_CUDACC_AT_LEAST(11, 7)

#if !defined(_CCCL_EXEC_CHECK_DISABLE)
#  if _CCCL_CUDA_COMPILER(NVCC)
#    define _CCCL_EXEC_CHECK_DISABLE _CCCL_PRAGMA(nv_exec_check_disable)
#  else
#    define _CCCL_EXEC_CHECK_DISABLE
#  endif // _CCCL_CUDA_COMPILER(NVCC)
#endif // !_CCCL_EXEC_CHECK_DISABLE

#endif // __CCCL_EXECUTION_SPACE_H
