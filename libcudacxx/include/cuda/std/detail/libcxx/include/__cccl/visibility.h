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

#include "../__cccl/compiler.h"
#include "../__cccl/system_header.h"

// We want to ensure that all warning emmiting from this header are supressed
#if defined(_CCCL_FORCE_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// For unknown reasons, nvc++ need to selectively disable this warning
// We do not want to use our usual macro because that would have push / pop semantics
#if defined(_CCCL_COMPILER_NVHPC)
#  pragma nv_diag_suppress 1407
#endif // _CCCL_COMPILER_NVHPC

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
