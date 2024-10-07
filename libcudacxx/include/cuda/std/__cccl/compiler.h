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

// Determine the host compiler and its version
#if defined(__INTEL_COMPILER)
#  define _CCCL_COMPILER_ICC
#elif defined(__NVCOMPILER)
#  define _CCCL_COMPILER_NVHPC
#  define _CCCL_COMPILER_NVHPC_VERSION \
    (__NVCOMPILER_MAJOR__ * 10000 + __NVCOMPILER_MINOR__ * 100 + __NVCOMPILER_PATCHLEVEL__)
#elif defined(__clang__)
#  define _CCCL_COMPILER_CLANG
#  define _CCCL_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif defined(__GNUC__)
#  define _CCCL_COMPILER_GCC
#  define _CCCL_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#  define _CCCL_COMPILER_MSVC
#  define _CCCL_MSVC_VERSION      _MSC_VER
#  define _CCCL_MSVC_VERSION_FULL _MSC_FULL_VER
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

// clang-cuda does not define __CUDACC_VER_MAJOR__ and friends. They are instead retrieved from the CUDA_VERSION macro
// defined in "cuda.h". clang-cuda automatically pre-includes "__clang_cuda_runtime_wrapper.h" which includes "cuda.h"
#if defined(_CCCL_CUDA_COMPILER_CLANG)
#  define _CCCL_CUDACC
#  define _CCCL_CUDACC_VER_MAJOR CUDA_VERSION / 1000
#  define _CCCL_CUDACC_VER_MINOR (CUDA_VERSION % 1000) / 10
#  define _CCCL_CUDACC_VER_BUILD 0
#  define _CCCL_CUDACC_VER       CUDA_VERSION * 100
#elif defined(_CCCL_CUDA_COMPILER)
#  define _CCCL_CUDACC
#  define _CCCL_CUDACC_VER_MAJOR __CUDACC_VER_MAJOR__
#  define _CCCL_CUDACC_VER_MINOR __CUDACC_VER_MINOR__
#  define _CCCL_CUDACC_VER_BUILD __CUDACC_VER_BUILD__
#  define _CCCL_CUDACC_VER       _CCCL_CUDACC_VER_MAJOR * 100000 + _CCCL_CUDACC_VER_MINOR * 1000 + _CCCL_CUDACC_VER_BUILD
#endif // _CCCL_CUDA_COMPILER

// Some convenience macros to filter CUDACC versions
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1102000)
#  define _CCCL_CUDACC_BELOW_11_2
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1102000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1103000)
#  define _CCCL_CUDACC_BELOW_11_3
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1103000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1104000)
#  define _CCCL_CUDACC_BELOW_11_4
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 11074000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1107000)
#  define _CCCL_CUDACC_BELOW_11_7
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1104000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1108000)
#  define _CCCL_CUDACC_BELOW_11_8
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1108000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1200000)
#  define _CCCL_CUDACC_BELOW_12_0
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1203000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1202000)
#  define _CCCL_CUDACC_BELOW_12_2
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1203000
#if !defined(_CCCL_CUDA_COMPILER) || (defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1203000)
#  define _CCCL_CUDACC_BELOW_12_3
#endif // defined(_CCCL_CUDACC) && _CCCL_CUDACC_VER < 1203000

#endif // __CCCL_COMPILER_H
