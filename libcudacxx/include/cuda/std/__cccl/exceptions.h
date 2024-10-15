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

#ifndef _CCCL_NO_EXCEPTIONS
#  if defined(CCCL_DISABLE_EXCEPTIONS) // Escape hatch for users to manually disable exceptions
#    define _CCCL_NO_EXCEPTIONS
#  elif defined(_CCCL_COMPILER_NVRTC) || (defined(_CCCL_COMPILER_MSVC) && _CPPUNWIND == 0) \
    || (!defined(_CCCL_COMPILER_MSVC) && !__EXCEPTIONS) // Catches all non msvc based compilers
#    define _CCCL_NO_EXCEPTIONS
#  endif
#endif // !_CCCL_NO_EXCEPTIONS

#endif // __CCCL_EXCEPTIONS_H
