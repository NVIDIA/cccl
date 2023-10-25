// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//


#ifndef _CUDA_PTX_PTX_ISA_TARGET_MACROS_H_
#define _CUDA_PTX_PTX_ISA_TARGET_MACROS_H_

#include <nv/target>            // __CUDA_MINIMUM_ARCH__ and friends

/*
 * Targeting macros
 *
 * Information from:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
 */


// SM version

#if (defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__) || (!defined(__CUDA_MINIMUM_ARCH__))
#  define __cccl_ptx_sm 900ULL
#elif (defined(__CUDA_MINIMUM_ARCH__) && 800 <= __CUDA_MINIMUM_ARCH__) || (!defined(__CUDA_MINIMUM_ARCH__))
#  define __cccl_ptx_sm 800ULL
// Fallback case. Define the SM version to be zero. This ensures that the macro is always defined.
#else
#  define __cccl_ptx_sm 0ULL
#endif


// PTX ISA version

// PTX ISA 8.3 is available from CTK 12.3, driver r545
#if   (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 3)) || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 830ULL
// PTX ISA 8.2 is available from CTK 12.2, driver r535
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 2)) || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 820ULL
// PTX ISA 8.1 is available from CTK 12.1, driver r530
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 1)) || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 810ULL
// PTX ISA 8.0 is available from CTK 12.0, driver r525
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 0)) || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 800ULL
// PTX ISA 7.8 is available from CTK 11.8, driver r520
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 8)) || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 780ULL
// Fallback case. Define the ISA version to be zero. This ensures that the macro is always defined.
#else
#  define __cccl_ptx_isa 0ULL
#endif

#endif // _CUDA_PTX_PTX_ISA_TARGET_MACROS_H_
