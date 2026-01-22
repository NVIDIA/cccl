//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_CUDA_TOOLKIT_H
#define __CCCL_CUDA_TOOLKIT_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/version.h>

#if _CCCL_CUDA_COMPILATION() || _CCCL_HAS_INCLUDE(<cuda_runtime_api.h>)
#  define _CCCL_HAS_CTK() 1
#else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#  define _CCCL_HAS_CTK() 0
#endif // ^^^ no cuda toolkit ^^^

// CUDA compilers preinclude cuda_runtime.h, so we need to include it here to get the CUDART_VERSION macro
#if _CCCL_HAS_CTK() && !_CCCL_CUDA_COMPILATION()
#  include <cuda_runtime_api.h>
#endif // _CCCL_HAS_CTK() && !_CCCL_CUDA_COMPILATION()

// Check compatibility of the CUDA compiler and CUDA toolkit headers
// Some users might want to use a newer version of the CTK than the compiler ships. Enable that on their own peril
#ifndef CCCL_DISABLE_CTK_COMPATIBILITY_CHECK
#  if _CCCL_CUDA_COMPILATION()
#    if !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#      error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
#    endif // !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#  endif // _CCCL_CUDA_COMPILATION()
#endif // CCCL_DISABLE_CTK_COMPATIBILITY_CHECK

#if _CCCL_HAS_CTK()
#  define _CCCL_CTK() (CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10)
#else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#  define _CCCL_CTK() _CCCL_VERSION_INVALID()
#endif // ^^^ no cuda toolkit ^^^

#define _CCCL_CTK_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)
#define _CCCL_CTK_BELOW(...)                   _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, <, __VA_ARGS__)
#define _CCCL_CTK_AT_LEAST(...)                _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, >=, __VA_ARGS__)

// Check CTK version compatibility.
#if _CCCL_HAS_CTK() && !defined(CCCL_ALLOW_UNSUPPORTED_CTK)
// CTK and CCCL versions are shifted by 10, for example CCCL 3.0 was released with CTK 13.0.
#  define _CCCL_MAJOR_TO_CTK_MAJOR_VERSION(_X) (_X + 10)

// CCCL supports whole previous CTK major release.
#  if _CCCL_CTK_BELOW(_CCCL_MAJOR_TO_CTK_MAJOR_VERSION(CCCL_MAJOR_VERSION) - 1, 0)
#    error \
      "This CCCL version does not support CUDA Toolkit below 12.0. Define CCCL_ALLOW_UNSUPPORTED_CTK to suppress this warning."
#  endif // _CCCL_CTK_BELOW(_CCCL_MAJOR_TO_CTK_MAJOR_VERSION(CCCL_MAJOR_VERSION) - 1, 0)

// CCCL is not forward compatible with CTK.
#  if _CCCL_CTK_AT_LEAST(_CCCL_MAJOR_TO_CTK_MAJOR_VERSION(CCCL_MAJOR_VERSION), CCCL_MINOR_VERSION + 1)
#    error \
      "Attempting to use CCCL with newer CUDA Toolkit than the version it was released with. CCCL is not forward compatible with future CUDA Toolkits. Define CCCL_ALLOW_UNSUPPORTED_CTK to suppress this warning."
#  endif // _CCCL_CTK_AT_LEAST(_CCCL_MAJOR_TO_CTK_MAJOR_VERSION(CCCL_MAJOR_VERSION), CCCL_MINOR_VERSION + 1)
#endif // _CCCL_HAS_CTK() && !CCCL_ALLOW_UNSUPPORTED_CTK

#endif // __CCCL_CUDA_TOOLKIT_H
