//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_EXCEPTIONS_H
#define __CCCL_EXCEPTIONS_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(CCCL_DISABLE_EXCEPTIONS) // Escape hatch for users to manually disable exceptions
#  define _CCCL_HAS_EXCEPTIONS() 0
#elif _CCCL_COMPILER(NVRTC) // NVRTC has no exceptions
#  define _CCCL_HAS_EXCEPTIONS() 0
#elif _CCCL_COMPILER(MSVC) // MSVC needs special checks for `_HAS_EXCEPTIONS` and `_CPPUNWIND`
#  define _CCCL_HAS_EXCEPTIONS() (_HAS_EXCEPTIONS != 0) && (_CPPUNWIND != 0)
#else // other compilers use `__EXCEPTIONS`
#  define _CCCL_HAS_EXCEPTIONS() __EXCEPTIONS
#endif // has exceptions

// The following macros are used to conditionally compile exception handling code. They
// are used in the same way as `try` and `catch` blocks, but they allow for different
// behavior based on whether exceptions are enabled or not, and whether the code is being
// compiled for device or not.
//
// Usage:
//   _CCCL_TRY({
//     ...                          // Code that may throw an exception
//   })
//   _CCCL_CATCH((cuda_error& e) {  // Handle CUDA exceptions
//     ...
//   })
//   _CCCL_CATCH((...) {            // Handle any other exceptions
//     ...
//   })
#if !_CCCL_HAS_EXCEPTIONS() || (_CCCL_CUDA_COMPILATION() && defined(__CUDA_ARCH__))
#  define _CCCL_TRY(...) {__VA_ARGS__}
#  define _CCCL_CATCH(...)
#elif _CCCL_CUDA_COMPILATION() && _CCCL_CUDA_COMPILER(NVHPC) // ^^^ no exceptions, or device code ^^^
                                                             // vvv CUDA compilation with NVHPC vvv
// In the following macro, it is intentional for the `else` branch to not have braces
// around the `try` block. This is to support the case where there is more than one
// `catch` clause.
#  define _CCCL_TRY(...)              \
    if target (nv::target::is_device) \
    {                                 \
      __VA_ARGS__                     \
    }                                 \
    else                              \
      try                             \
      {                               \
        __VA_ARGS__                   \
      }
#  define _CCCL_CATCH(...) catch __VA_ARGS__
#else // ^^^ CUDA compilation with NVHPC ^^^
      // vvv Host compilation with exceptions vvv
#  define _CCCL_TRY(...) \
    try                  \
    {                    \
      __VA_ARGS__        \
    }
#  define _CCCL_CATCH(...) catch __VA_ARGS__
#endif // ^^^ Host compilation with exceptions ^^^

#endif // __CCCL_EXCEPTIONS_H
