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
#include <cuda/std/__cccl/execution_space.h>
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
#  define _CCCL_HAS_EXCEPTIONS() ((_HAS_EXCEPTIONS != 0) && (_CPPUNWIND != 0)) // disabled with /EH
#else // other compilers use `__EXCEPTIONS`
#  define _CCCL_HAS_EXCEPTIONS() (__EXCEPTIONS) // disabled with -fno-exceptions
#endif // has exceptions

#endif // __CCCL_EXCEPTIONS_H
