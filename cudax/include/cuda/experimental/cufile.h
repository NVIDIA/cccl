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

// ================================================================================================
// Core Components
// ================================================================================================

#include <cuda/experimental/__cufile/cufile.hpp>

#endif // __CUDAX_CUFILE_H