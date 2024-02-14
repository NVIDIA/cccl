//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_COMPILER_H
#define __CCCL_COMPILER_H

// Determine the host compiler
#if defined(__INTEL_LLVM_COMPILER)
#  define _CCCL_COMPILER_ICC_LLVM
#elif defined(__INTEL_COMPILER)
#  define _CCCL_COMPILER_ICC
#elif defined(__NVCOMPILER)
#  define _CCCL_COMPILER_NVHPC
#elif defined(__clang__)
#  define _CCCL_COMPILER_CLANG
#elif defined(__GNUC__)
#  define _CCCL_COMPILER_GCC
#elif defined(_MSC_VER)
#  define _CCCL_COMPILER_MSVC
#elif defined(__IBMCPP__)
#  define _CCCL_COMPILER_IBM
#elif defined(__CUDACC_RTC__)
#  define _CCCL_COMPILER_NVRTC
#endif

// Convenient shortcut to determine which version of MSVC we are dealing with
#if defined(_CCCL_COMPILER_MSVC)
#  if _MSC_VER < 1920
#    define _CCCL_COMPILER_MSVC_2017
#  elif _MSC_VER < 1930
#    define _CCCL_COMPILER_MSVC_2019
#  else // _MSC_VER < 1940
#    define _CCCL_COMPILER_MSVC_2022
#  endif // _MSC_VER < 1940
#endif // _CCCL_COMPILER_MSVC

// Determine the cuda compiler
#if defined(__NVCC__)
#  define _CCCL_CUDA_COMPILER_NVCC
#elif defined(_NVHPC_CUDA)
#  define _CCCL_CUDA_COMPILER_NVHPC
#elif defined(__CUDA__) && defined(_CCCL_COMPILER_CLANG)
#  define _CCCL_CUDA_COMPILER_CLANG
#endif

// Shorthand to check whether there is a cuda compiler available
#if defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_CUDA_COMPILER_NVHPC) || defined(_CCCL_CUDA_COMPILER_CLANG) \
  || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_CUDA_COMPILER
#endif // cuda compiler available

#endif // __CCCL_COMPILER_H
