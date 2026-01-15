//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_CUDA_CAPABILITIES
#define __CCCL_CUDA_CAPABILITIES

#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/cuda_toolkit.h>

#include <nv/target>

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:
//! When this macro is defined, Programmatic Dependent Launch (PDL) is disabled across CCCL
#  define CCCL_DISABLE_PDL
#endif // _CCCL_DOXYGEN_INVOKED

#ifdef CCCL_DISABLE_PDL
#  define _CCCL_HAS_PDL() 0
#else // CCCL_DISABLE_PDL
#  define _CCCL_HAS_PDL() 1
#endif // CCCL_DISABLE_PDL

#if _CCCL_HAS_PDL()
// Waits for the previous kernel to complete (when it reaches its final membar). Should be put before the first global
// memory access in a kernel.
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC() NV_IF_TARGET(NV_PROVIDES_SM_90, ::cudaGridDependencySynchronize();)
// Allows the subsequent kernel in the same stream to launch. Can be put anywhere in a kernel.
// Heuristic(ahendriksen): put it after the last load.
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH() NV_IF_TARGET(NV_PROVIDES_SM_90, ::cudaTriggerProgrammaticLaunchCompletion();)
#else // _CCCL_HAS_PDL()
#  define _CCCL_PDL_GRID_DEPENDENCY_SYNC()
#  define _CCCL_PDL_TRIGGER_NEXT_LAUNCH()
#endif // _CCCL_HAS_PDL()

// Check whether the relocatable device code (RDC) is being generated.
#if defined(__CUDACC_RDC__) || defined(__CLANG_RDC__) || defined(_NVHPC_RDC)
#  define _CCCL_HAS_RDC() 1
#else // ^^^ has RDC ^^^ / vvv no RDC vvv
#  define _CCCL_HAS_RDC() 0
#endif // ^^^ no RDC ^^^

// Check whether extensible whole program is being compiled.
#if defined(__CUDACC_EWP__)
#  define _CCCL_HAS_EWP() 1
#else // ^^^ has EWP ^^^ / vvv no EWP vvv
#  define _CCCL_HAS_EWP() 0
#endif // ^^^ no EWP ^^^

// Control whether device runtime APIs can be used, because they require libcudadevrt to be linked. Defaults to true
// when RDC or EWP are enabled. Can be disabled by defining CCCL_DISABLE_DEVICE_RUNTIME.
#if (_CCCL_HAS_RDC() || _CCCL_HAS_EWP()) && !defined(CCCL_DISABLE_DEVICE_RUNTIME)
#  define _CCCL_HAS_DEVICE_RUNTIME() 1
#else // ^^^ has device runtime ^^^ / vvv no device runtime vvv
#  define _CCCL_HAS_DEVICE_RUNTIME() 0
#endif // ^^^ no device runtime ^^^

// Some functions can be called from host or device code and launch kernels inside. Thus, they use CUDA Dynamic
// Parallelism (CDP) and require compiling with Relocatable Device Code (RDC) or extensible whole program (EWP) and link
// with device runtime library. CDP is unsupported with clang-cuda below 22.
// TODO(bgruber): remove CUB_DISABLE_CDP in CCCL 4.0
#if _CCCL_HAS_DEVICE_RUNTIME() && !defined(CCCL_DISABLE_CDP) && !defined(CUB_DISABLE_CDP) \
  && !_CCCL_CUDA_COMPILER(CLANG, <, 22)
// We have CDP, so host and device APIs can call kernels
#  define _CCCL_HAS_CDP() 1
#else // ^^^ has CDP ^^^ / vvv no CDP vvv
// We don't have CDP, only host APIs can call kernels
#  define _CCCL_HAS_CDP() 0
#endif // ^^^ no CDP ^^^

#if _CCCL_HAS_CDP()
#  ifdef CUDA_FORCE_CDP1_IF_SUPPORTED
#    error "CUDA Dynamic Parallelism 1 is no longer supported. Please undefine CUDA_FORCE_CDP1_IF_SUPPORTED."
#  endif // CUDA_FORCE_CDP1_IF_SUPPORTED
#endif // _CCCL_HAS_CDP()

#endif // __CCCL_CUDA_CAPABILITIES
