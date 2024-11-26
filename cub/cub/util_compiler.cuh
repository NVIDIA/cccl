/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Detect compiler information.
 */

#pragma once

// For _CCCL_IMPLICIT_SYSTEM_HEADER
#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// enumerate host compilers we know about
//! deprecated [Since 2.7]
#define CUB_HOST_COMPILER_UNKNOWN 0
//! deprecated [Since 2.7]
#define CUB_HOST_COMPILER_MSVC 1
//! deprecated [Since 2.7]
#define CUB_HOST_COMPILER_GCC 2
//! deprecated [Since 2.7]
#define CUB_HOST_COMPILER_CLANG 3

// enumerate device compilers we know about
//! deprecated [Since 2.7]
#define CUB_DEVICE_COMPILER_UNKNOWN 0
//! deprecated [Since 2.7]
#define CUB_DEVICE_COMPILER_MSVC 1
//! deprecated [Since 2.7]
#define CUB_DEVICE_COMPILER_GCC 2
//! deprecated [Since 2.7]
#define CUB_DEVICE_COMPILER_NVCC 3
//! deprecated [Since 2.7]
#define CUB_DEVICE_COMPILER_CLANG 4

// figure out which host compiler we're using
#if _CCCL_COMPILER(MSVC)
//! deprecated [Since 2.7]
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_MSVC
//! deprecated [Since 2.7]
#  define CUB_MSVC_VERSION _MSC_VER
//! deprecated [Since 2.7]
#  define CUB_MSVC_VERSION_FULL _MSC_FULL_VER
#elif _CCCL_COMPILER(CLANG)
//! deprecated [Since 2.7]
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_CLANG
//! deprecated [Since 2.7]
#  define CUB_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#elif _CCCL_COMPILER(GCC)
//! deprecated [Since 2.7]
#  define CUB_HOST_COMPILER CUB_HOST_COMPILER_GCC
//! deprecated [Since 2.7]
#  define CUB_GCC_VERSION   (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

// figure out which device compiler we're using
#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(NVHPC)
//! deprecated [Since 2.7]
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_NVCC
#elif _CCCL_COMPILER(MSVC)
//! deprecated [Since 2.7]
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_MSVC
#elif _CCCL_COMPILER(GCC)
//! deprecated [Since 2.7]
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_GCC
#elif _CCCL_COMPILER(CLANG)
// CUDA-capable clang should behave similar to NVCC.
#  if _CCCL_CUDA_COMPILER(NVCC)
//! deprecated [Since 2.7]
#    define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_NVCC
#  else
//! deprecated [Since 2.7]
#    define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_CLANG
#  endif
#else
//! deprecated [Since 2.7]
#  define CUB_DEVICE_COMPILER CUB_DEVICE_COMPILER_UNKNOWN
#endif
