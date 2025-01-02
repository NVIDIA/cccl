//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_OS_H
#define __CCCL_OS_H

// The header provides the following macros to determine the host architecture:
//
// _CCCL_OS(WINDOWS)
// _CCCL_OS(LINUX)
// _CCCL_OS(ANDROID)
// _CCCL_OS(QNX)

// Determine the host compiler and its version
#if defined(_WIN32)
#  define _CCCL_OS_WINDOWS_() 1
#elif defined(__linux__)
#  define _CCCL_OS_LINUX_() 1
#endif

#if defined(__ANDROID__)
#  define _CCCL_OS_ANDROID_() 1
#elif defined(__QNX__) || defined(__QNXNTO__)
#  define _CCCL_OS_QNX_() 1
#endif

#define _CCCL_OS(...) _CCCL_OS_##__VA_ARGS__##_()

#endif // __CCCL_OS_H
