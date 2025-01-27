//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_ARCH_H
#define __CCCL_ARCH_H

// The header provides the following macros to determine the host architecture:
//
// _CCCL_ARCH(ARM64)     ARM64
// _CCCL_ARCH(X86_64)    X86 64 bit

// Determine the host compiler and its version

// Arm 64-bit
#if (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) /*emulation*/)
#  define _CCCL_ARCH_ARM64_() 1
#else
#  define _CCCL_ARCH_ARM64_() 0
#endif

// X86 64-bit

// _M_X64 is defined even if we are compiling in Arm64 emulation mode
#if (defined(_M_X64) && !defined(_M_ARM64EC)) || defined(__amd64__) || defined(__x86_64__)
#  define _CCCL_ARCH_X86_64_() 1
#else
#  define _CCCL_ARCH_X86_64_() 0
#endif

#define _CCCL_ARCH(...) _CCCL_ARCH_##__VA_ARGS__##_()

#endif // __CCCL_ARCH_H
