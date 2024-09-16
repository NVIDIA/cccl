/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file compiler.h
 *  \brief Compiler-specific configuration
 */

#pragma once

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// enumerate host compilers we know about
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_UNKNOWN 0
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_MSVC 1
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_GCC 2
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_CLANG 3
//! deprecated [Since 2.7]
#define THRUST_HOST_COMPILER_INTEL 4

// enumerate device compilers we know about
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_UNKNOWN 0
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_MSVC 1
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_GCC 2
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_CLANG 3
//! deprecated [Since 2.7]
#define THRUST_DEVICE_COMPILER_NVCC 4

// figure out which host compiler we're using
#if defined(_CCCL_COMPILER_MSVC)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_MSVC
//! deprecated [Since 2.7]
#  define THRUST_MSVC_VERSION _CCCL_MSVC_VERSION
//! deprecated [Since 2.7]
#  define THRUST_MSVC_VERSION_FULL _CCCL_MSVC_VERSION_FULL
#elif defined(_CCCL_COMPILER_ICC)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_INTEL
#elif defined(_CCCL_COMPILER_CLANG)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_CLANG
//! deprecated [Since 2.7]
#  define THRUST_CLANG_VERSION _CCCL_CLANG_VERSION
#elif defined(_CCCL_COMPILER_GCC)
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_GCC
//! deprecated [Since 2.7]
#  define THRUST_GCC_VERSION   _CCCL_GCC_VERSION
#  if (THRUST_GCC_VERSION >= 50000)
//! deprecated [Since 2.7]
#    define THRUST_MODERN_GCC
#  else
//! deprecated [Since 2.7]
#    define THRUST_LEGACY_GCC
#  endif
#else
//! deprecated [Since 2.7]
#  define THRUST_HOST_COMPILER THRUST_HOST_COMPILER_UNKNOWN
#endif // TRUST_HOST_COMPILER

// figure out which device compiler we're using
#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#elif defined(_CCCL_COMPILER_MSVC)
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_MSVC
#elif defined(_CCCL_COMPILER_GCC)
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_GCC
#elif defined(_CCCL_COMPILER_CLANG)
// CUDA-capable clang should behave similar to NVCC.
#  if defined(__CUDA__)
//! deprecated [Since 2.7]
#    define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_NVCC
#  else
//! deprecated [Since 2.7]
#    define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_CLANG
#  endif
#else
//! deprecated [Since 2.7]
#  define THRUST_DEVICE_COMPILER THRUST_DEVICE_COMPILER_UNKNOWN
#endif

// is the device compiler capable of compiling omp?
#if defined(_OPENMP) || defined(_NVHPC_STDPAR_OPENMP)
#  define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_TRUE
#else
#  define THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE THRUST_FALSE
#endif // _OPENMP
