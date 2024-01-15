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

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#  define _CCCL_HOST         __host__
#  define _CCCL_DEVICE       __device__
#  define _CCCL_HOST_DEVICE  __host__ __device__
#  define _CCCL_FORCEINLINE __forceinline__
#else // ^^^ __CUDACC__ || _NVHPC_CUDA ^^^ / vvv !__CUDACC__ && !_NVHPC_CUDA
#  define _CCCL_HOST
#  define _CCCL_DEVICE
#  define _CCCL_HOST_DEVICE
#  define _CCCL_FORCEINLINE
#endif // !__CUDACC__ && !_NVHPC_CUDA

#if !defined(_CCCL_EXEC_CHECK_DISABLE)
#  if defined(_CCCL_COMPILER_NVRTC) || defined(_CCCL_CUDA_COMPILER_NVHPC) || defined(_CCCL_CUDA_COMPILER_CLANG)
#    define _CCCL_EXEC_CHECK_DISABLE
#  else // defined(_CCCL_CUDA_COMPILER_NVCC)
#    if defined(_CCCL_COMPILER_MSVC)
#      define _CCCL_EXEC_CHECK_DISABLE __pragma("nv_exec_check_disable")
#    else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv !_CCCL_COMPILER_MSVC vvv
#      define _CCCL_EXEC_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#    endif // !_CCCL_COMPILER_MSVC
#  endif // _CCCL_CUDA_COMPILER_NVCC
#endif // !_CCCL_EXEC_CHECK_DISABLE

#endif // __CCCL_EXECUTION_SPACE_H
