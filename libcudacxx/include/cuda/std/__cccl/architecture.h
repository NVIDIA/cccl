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

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/preprocessor.h>

// The header provides the following macros to determine the host architecture:
//
// _CCCL_ARCH(ARM64)     ARM64
// _CCCL_ARCH(X86_64)    X86 64 bit
// CCCL_ARCH(ARM64)      ARM64
// CCCL_ARCH(X86_64)     X86 64 bit

// Determine the host architecture

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

//! @def CCCL_ARCH(ARCH) /* implementation defined */
//!
//! @brief Detect the current host architecture.
//!
//! @param ARCH The name of the host architecture to test.
//!
//! @note This macro is made available when including any libcu++ header. Users that wish to
//! include the smallest possible header for this macro should include `<cuda/std/version>`.
//!
//! For supported host architectures, the macro expands to an implementation-defined true value
//! if the current host architecture matches, or false otherwise. These values may be used in
//! boolean expressions (preprocessor or otherwise), but no other guarantees are made.
//!
//! Available values for `ARCH` include:
//!
//! - ``ARM64``: ARM 64-bit, including MSVC ARM64EC emulation.
//! - ``X86_64``: X86 64-bit. This is false when compiling in MSVC ARM64EC emulation mode.
//!
//! Passing any other value will result in an undefined expansion, which may or may not be
//! diagnosed by the compiler.
//!
//! @par Example
//! @code
//! #define MY_OTHER_MACRO 1
//!
//! // Expansion value can be used in ordinary macro conditionals
//! #if CCCL_ARCH(X86_64) && MY_OTHER_MACRO
//!   // ...
//! #endif
//!
//! // Can be negated as usual
//! #if !CCCL_ARCH(ARM64)
//!   // ...
//! #endif
//! @endcode
//!
//! @return true if the specified host architecture is being compiled for, false otherwise.
#ifdef _CCCL_DOXYGEN_INVOKED
#  define CCCL_ARCH(ARCH) /* implementation defined */
#else
#  define CCCL_ARCH(__arch__) _CCCL_ARCH_##__arch__##_()
#endif

// Note: the public API is single-arg to constrain the API and allow for future expansion. The
// implementation is duplicated to guard against the architecture targets being accidentally
// defined by the user.

// Determine the endianness

#define _CCCL_ENDIAN_LITTLE() 0xDEAD
#define _CCCL_ENDIAN_BIG()    0xFACE
#define _CCCL_ENDIAN_PDP()    0xBEEF

#if _CCCL_COMPILER(NVRTC) || (_CCCL_COMPILER(MSVC) && (_CCCL_ARCH(X86_64) || _CCCL_ARCH(ARM64))) || __LITTLE_ENDIAN__
#  define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_LITTLE()
#elif __BIG_ENDIAN__
#  define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_BIG()
#elif defined(__BYTE_ORDER__)
#  if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#    define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_LITTLE()
#  elif __BYTE_ORDER__ == __ORDER_PDP_ENDIAN__
#    define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_PDP()
#  elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#    define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_BIG()
#  endif // __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#elif __has_include(<endian.h>)
#  include <endian.h>
#  if __BYTE_ORDER == __LITTLE_ENDIAN
#    define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_LITTLE()
#  elif __BYTE_ORDER == __PDP_ENDIAN
#    define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_PDP()
#  elif __BYTE_ORDER == __BIG_ENDIAN
#    define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_BIG()
#  endif // __BYTE_ORDER == __BIG_ENDIAN
#endif // ^^^ has endian.h ^^^

#if !defined(_CCCL_ENDIAN_NATIVE)
_CCCL_WARNING("failed to determine the endianness of the host architecture, defaulting to little-endian")
#  define _CCCL_ENDIAN_NATIVE() _CCCL_ENDIAN_LITTLE()
#endif // !_CCCL_ENDIAN_NATIVE

#define _CCCL_ENDIAN(_NAME) (_CCCL_ENDIAN_NATIVE() == _CCCL_ENDIAN_##_NAME())

#endif // __CCCL_ARCH_H
