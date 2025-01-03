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
// _CCCL_ARCH(X86)       X86 both 32 and 64 bit
// _CCCL_ARCH(X86_64)    X86 64 bit
// _CCCL_ARCH(X86_32)    X86 64 bit
// _CCCL_ARCH(64BIT)     Any 64 bit OS (supported by CUDA)
// _CCCL_ARCH(32BIT)     Any 32 bit OS (supported by CUDA)

// Determine the host compiler and its version
#define _CCCL_ARCH_ARM64_()  (defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC) /*emulation*/)
#define _CCCL_ARCH_X86_64_() defined(_M_X64) || defined(__amd64__) || defined(__x86_64__)
#define _CCCL_ARCH_X86_32_() defined(_M_IX86)
#define _CCCL_ARCH_X86_()    (_CCCL_ARCH_X86_64_() || _CCCL_ARCH_X86_32_())
#define _CCCL_ARCH_64BIT_()  (_CCCL_ARCH_ARM64_() || _CCCL_ARCH_X86_64_())
#define _CCCL_ARCH_32BIT_()  defined(_M_IX86)

#define _CCCL_ARCH(...) _CCCL_ARCH_##__VA_ARGS__##_()

#endif // __CCCL_ARCH_H
