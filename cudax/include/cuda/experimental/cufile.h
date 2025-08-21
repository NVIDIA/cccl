//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef __CUDAX_CUFILE_H
#define __CUDAX_CUFILE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file cufile.h
 * @brief Modern C++ bindings for NVIDIA cuFILE (GPU Direct Storage)
 *
 * Provides clean, modern C++ interface that directly maps to the cuFILE C API.
 */

// Feature detection: Check if the C cuFILE headers are available
#ifndef CUDAX_HAS_CUFILE
#    if _CCCL_HAS_INCLUDE(<cufile.h>)
#      define _CUDAX_HAS_CUFILE() 1
#    else
#      define CUDAX_HAS_CUFILE 0
#    endif
#  else
#    define CUDAX_HAS_CUFILE 0
#  endif
#endif // CUDAX_HAS_CUFILE

// ================================================================================================
// Core Components
// ================================================================================================

#if CUDAX_HAS_CUFILE
#  include <cuda/experimental/__cufile/cufile.hpp>
#else
// cuFILE not available on this platform. The header is safe to include, but
// no cuFILE APIs are provided. Use CUDAX_HAS_CUFILE to conditionally compile code.
namespace cuda::experimental::cufile
{
} // namespace [cuda](cuda::experimental::cufile)
#endif

#endif // __CUDAX_CUFILE_H
