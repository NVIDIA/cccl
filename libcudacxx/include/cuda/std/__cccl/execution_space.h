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

#ifndef __has_include
#  define __has_include(x) 0
#endif // !__has_include

// Bring in the __host__ and __device__ macros:
#if __has_include(<crt/host_defines.h>)
#  define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#  include <crt/host_defines.h>
#endif

#if defined(_CCCL_CUDA_COMPILER)
#  define _CCCL_HOST        __host__
#  define _CCCL_DEVICE      __device__
#  define _CCCL_HOST_DEVICE __host__ __device__
#  define _CCCL_FORCEINLINE __forceinline__
#else // ^^^ _CCCL_CUDA_COMPILER ^^^ / vvv !_CCCL_CUDA_COMPILER
#  define _CCCL_HOST
#  define _CCCL_DEVICE
#  define _CCCL_HOST_DEVICE
#  define _CCCL_FORCEINLINE inline
#endif // !_CCCL_CUDA_COMPILER

#if !defined(_CCCL_EXEC_CHECK_DISABLE)
#  if defined(_CCCL_CUDA_COMPILER_NVCC)
#    if defined(_CCCL_COMPILER_MSVC)
#      define _CCCL_EXEC_CHECK_DISABLE __pragma("nv_exec_check_disable")
#    else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#      define _CCCL_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#    endif // !_CCCL_COMPILER_MSVC
#  else
#    define _CCCL_EXEC_CHECK_DISABLE
#  endif // _CCCL_CUDA_COMPILER_NVCC
#endif // !_CCCL_EXEC_CHECK_DISABLE

#endif // __CCCL_EXECUTION_SPACE_H
