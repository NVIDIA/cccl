//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_PTX_ISA_H_
#define __CCCL_PTX_ISA_H_

#include "../__cccl/compiler.h"
#include "../__cccl/system_header.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

/*
 * Targeting macros
 *
 * Information from:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
 */

// PTX ISA 8.3 is available from CUDA 12.3, driver r545
// The first define is for future major versions of CUDACC.
// We make sure that these get the highest known PTX ISA version.
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 12)) || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 830ULL
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 3)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 830ULL
// PTX ISA 8.2 is available from CUDA 12.2, driver r535
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 2)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 820ULL
// PTX ISA 8.1 is available from CUDA 12.1, driver r530
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 1)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 810ULL
// PTX ISA 8.0 is available from CUDA 12.0, driver r525
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 0)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 800ULL
// PTX ISA 7.8 is available from CUDA 11.8, driver r520
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 8)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 780ULL
// PTX ISA 7.7 is available from CUDA 11.7, driver r515
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 7)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 770ULL
// PTX ISA 7.6 is available from CUDA 11.6, driver r510
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 6)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 760ULL
// PTX ISA 7.5 is available from CUDA 11.5, driver r495
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 5)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 750ULL
// PTX ISA 7.4 is available from CUDA 11.4, driver r470
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 740ULL
// PTX ISA 7.3 is available from CUDA 11.3, driver r465
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 3)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 730ULL
// PTX ISA 7.2 is available from CUDA 11.2, driver r460
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 2)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 720ULL
// PTX ISA 7.1 is available from CUDA 11.1, driver r455
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 1)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 710ULL
// PTX ISA 7.0 is available from CUDA 11.0, driver r445
#elif (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 0)) \
  || (!defined(__CUDACC_VER_MAJOR__))
#  define __cccl_ptx_isa 700ULL
// Fallback case. Define the ISA version to be zero. This ensures that the macro is always defined.
#else
#  define __cccl_ptx_isa 0ULL
#endif

// We define certain feature test macros depending on availability. When
// __CUDA_MINIMUM_ARCH__ is not available, we define the following features
// depending on PTX ISA. This permits checking for the feature in host code.
// When __CUDA_MINIMUM_ARCH__ is available, we only enable the feature when the
// hardware supports it.
#if __cccl_ptx_isa >= 800
#if (!defined(__CUDA_MINIMUM_ARCH__)) \
  || (defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__)
#  define __cccl_lib_local_barrier_arrive_tx
#  define __cccl_lib_experimental_ctk12_cp_async_exposure
#endif
#endif // __cccl_ptx_isa >= 800

// NVRTC uses its own <nv/target> header, so we need to manually tell it when we expect SM90a to be available
#if defined(_CCCL_COMPILER_NVRTC) && !defined(NV_HAS_FEATURE_SM_90a)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL))
#define NV_HAS_FEATURE_SM_90a NV_PROVIDES_SM_90
#else // ^^^ SM90a ^^^ / vvv !SM90a vvv
#define NV_HAS_FEATURE_SM_90a NV_NO_TARGET
#endif //
#endif // _CCCL_COMPILER_NVRTC && !NV_HAS_FEATURE_SM_90a

#endif // __CCCL_PTX_ISA_H_
