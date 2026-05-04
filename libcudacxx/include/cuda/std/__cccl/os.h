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
#if defined(_WIN32) || defined(_WIN64) /* _WIN64 for NVRTC */
#  define _CCCL_OS_WINDOWS_() 1
#else
#  define _CCCL_OS_WINDOWS_() 0
#endif

#if defined(__linux__) || defined(__LP64__) /* __LP64__ for NVRTC */
#  define _CCCL_OS_LINUX_() 1
#else
#  define _CCCL_OS_LINUX_() 0
#endif

#if defined(__ANDROID__)
#  define _CCCL_OS_ANDROID_() 1
#else
#  define _CCCL_OS_ANDROID_() 0
#endif

#if defined(__QNX__) || defined(__QNXNTO__)
#  define _CCCL_OS_QNX_() 1
#else
#  define _CCCL_OS_QNX_() 0
#endif

#if defined(__APPLE__) || defined(__APPLE_CC__)
#  define _CCCL_OS_APPLE_() 1
#else
#  define _CCCL_OS_APPLE_() 0
#endif

#define _CCCL_OS(...) _CCCL_OS_##__VA_ARGS__##_()

//! @brief Detect the current operating system.
//!
//! @param __os__ The name of the operating system to test.
//!
//! @note This macro is made available when including any libcu++ header. Users that wish to
//! include the smallest possible header for this macro should include `<cuda/std/version>`.
//!
//! For supported operating systems, the macro expands to an implementation-defined true value
//! if the current operating system matches, or false otherwise. These values may be used in
//! boolean expressions (preprocessor or otherwise), but no other guarantees are made.
//!
//! Available values for `__os__` include:
//!
//! - ``WINDOWS``: Windows, either in 32-bit or 64-bit mode.
//! - ``LINUX``: Any kind of Linux installation. Note that other unix-based operating systems will
//!              also match against this.
//! - ``ANDROID``: Android operating system.
//! - ``QNX``: QNX real-time operating system.
//! - ``APPLE``: macOS (Intel or Apple Silicon).
//!
//! Passing any other value will result in an undefined expansion, which may or may not be
//! diagnosed by the compiler.
//!
//! @note Some operating systems may satisfy multiple conditions. For example macOS and Android
//! satisfy both `APPLE`/`ANDROID` and `LINUX`.
//!
//! @par Example
//! @code
//! #define MY_OTHER_MACRO 1
//!
//! // Expansion value can be used in ordinary macro conditionals
//! #if CCCL_OS(WINDOWS) && MY_OTHER_MACRO
//!   // ...
//! #endif
//!
//! // Can be negated as usual
//! #if !CCCL_OS(QNX)
//!   // ...
//! #endif
//!
//! #if CCCL_OS(APPLE)
//!   // Will be visible only on macOS
//! #endif
//!
//! #if CCCL_OS(ANDROID)
//!   // Will be visible only on Android
//! #endif
//!
//! #if CCCL_OS(LINUX) && !CCCL_OS(APPLE) && !CCCL_OS(ANDROID)
//!   // Only visible on Linux
//! #endif
//! @endcode
//!
//! @return true if the specified OS is begin compiled for, false otherwise.
#define CCCL_OS(__os__) _CCCL_OS_##__os__##_()

// Note: the public API is single-arg to constrain the API and allow for future expansion. The
// implementation is duplicated to guard against the OS targets being accidentally defined by
// the user.

#endif // __CCCL_OS_H
