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


#ifndef PTX_ISA_TARGET_MACROS_H_
#define PTX_ISA_TARGET_MACROS_H_


/*
 * Targeting macros
 *
 * Information from:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
 */

#if (defined(__CUDA_MINIMUM_ARCH__) && 800 <= __CUDA_MINIMUM_ARCH__) || (!defined(__CUDA_MINIMUM_ARCH__))
#  define _LIBCUDACXX_PTX_SM_80_AVAILABLE
#endif

#if (defined(__CUDA_MINIMUM_ARCH__) && 900 <= __CUDA_MINIMUM_ARCH__) || (!defined(__CUDA_MINIMUM_ARCH__))
#  define _LIBCUDACXX_PTX_SM_90_AVAILABLE
#endif

// PTX ISA 7.8 is available from CTK 11.8, driver r520
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 8)) || (!defined(__CUDACC_VER_MAJOR__))
#  define _LIBCUDACXX_PTX_ISA_78_AVAILABLE
#endif

// PTX ISA 7.8 is available from CTK 11.8, driver r520 (so also from CTK 12.0 onwards)
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 0)) || (!defined(__CUDACC_VER_MAJOR__))
#  define _LIBCUDACXX_PTX_ISA_78_AVAILABLE
#endif

// PTX ISA 8.0 is available from CTK 12.0, driver r525
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 0)) || (!defined(__CUDACC_VER_MAJOR__))
#  define _LIBCUDACXX_PTX_ISA_80_AVAILABLE
#endif

// PTX ISA 8.1 is available from CTK 12.1, driver r530
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 1)) || (!defined(__CUDACC_VER_MAJOR__))
#  define _LIBCUDACXX_PTX_ISA_81_AVAILABLE
#endif

// PTX ISA 8.2 is available from CTK 12.2, driver r535
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 2)) || (!defined(__CUDACC_VER_MAJOR__))
#  define _LIBCUDACXX_PTX_ISA_82_AVAILABLE
#endif

// PTX ISA 8.3 is available from CTK 12.3, driver r545
#if (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 3)) || (!defined(__CUDACC_VER_MAJOR__))
#  define _LIBCUDACXX_PTX_ISA_83_AVAILABLE
#endif


#endif // PTX_ISA_TARGET_MACROS_H_
