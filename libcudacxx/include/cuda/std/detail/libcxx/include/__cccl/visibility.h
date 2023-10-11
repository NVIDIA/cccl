//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_VISIBILITY_H
#define __CCCL_VISIBILITY_H

#ifndef __CCCL_CONFIG
#error "<__cccl/visibility.h> should only be included in from <cuda/__cccl_config>"
#endif // __CCCL_CONFIG

// We want to ensure that all warning emmiting from this header are supressed
// We define cub and thrust kernels as hidden. However, this triggers errors about missing external linkage iff the
// definition of the _CCCL_ATTRIBUTE_HIDDEN macro is not in a system header :shrug:
// FIXME: this currently breaks nvc++
#if defined(_CCCL_COMPILER_NVHPC)
_CCCL_IMPLICIT_SYSTEM_HEADER
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_FORCE_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

// Enable us to hide kernels
#if defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_ATTRIBUTE_HIDDEN
#elif defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_ATTRIBUTE_HIDDEN
#else // ^^^ _CCCL_COMPILER_NVRTC ^^^ / vvv _CCCL_COMPILER_NVRTC vvv
#  define _CCCL_ATTRIBUTE_HIDDEN __attribute__ ((__visibility__("hidden")))
#endif // !_CCCL_COMPILER_NVRTC

#if !defined(CCCL_DETAIL_KERNEL_ATTRIBUTES)
#  define CCCL_DETAIL_KERNEL_ATTRIBUTES __global__ _CCCL_ATTRIBUTE_HIDDEN
#endif // !CCCL_DETAIL_KERNEL_ATTRIBUTES

#endif // __CCCL_VISIBILITY_H
