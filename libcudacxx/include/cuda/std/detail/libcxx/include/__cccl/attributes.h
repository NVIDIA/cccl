//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_ATTRIBUTES_H
#define __CCCL_ATTRIBUTES_H

#include <cuda/std/detail/libcxx/include/__cccl/compiler.h>
#include <cuda/std/detail/libcxx/include/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#ifndef __has_cpp_attribute
#  define __has_cpp_attribute(__x) 0
#endif // !__has_cpp_attribute

#if __has_cpp_attribute(nodiscard) || (defined(_CCCL_COMPILER_MSVC) && _CCCL_STD_VER >= 2017)
#  define _CCCL_NODISCARD [[nodiscard]]
#else // ^^^ has nodiscard ^^^ / vvv no nodiscard vvv
#  define _CCCL_NODISCARD
#endif // no nodiscard

// NVCC below 11.3 does not support nodiscard on friend operators
// It always fails with clang
#if defined(_CCCL_CUDACC_BELOW_11_3) || defined(_CCCL_COMPILER_CLANG)
#  define _CCCL_NODISCARD_FRIEND friend
#else // ^^^ _CCCL_CUDACC_BELOW_11_3 ^^^ / vvv !_CCCL_CUDACC_BELOW_11_3 vvv
#  define _CCCL_NODISCARD_FRIEND _CCCL_NODISCARD friend
#endif // !_CCCL_CUDACC_BELOW_11_3 && !_CCCL_COMPILER_CLANG

#endif // __CCCL_ATTRIBUTES_H
