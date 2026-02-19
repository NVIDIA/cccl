//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_AUTO_CAST_H
#define _CUDA_STD___UTILITY_AUTO_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>

#if _CCCL_COMPILER(GCC, >=, 12) || _CCCL_COMPILER(NVHPC)
#  define _CCCL_HAS_AUTO_CAST_BEFORE_CXX23() 1
#else // ^^^ has auto(expr) before c++23 ^^^ / vvv no auto(expr) before c++23 vvv
#  define _CCCL_HAS_AUTO_CAST_BEFORE_CXX23() 0
#endif // ^^^ no auto(expr) before c++23 ^^^

// nvcc < 13 does not support auto(expr), nvhpc fails to use auto(expr) in noexcept(noexcept(...)) (nvbug 5742468)
#if _CCCL_CUDA_COMPILER(NVCC, <, 13) || _CCCL_COMPILER(NVHPC)
#  undef _CCCL_HAS_AUTO_CAST_BEFORE_CXX23
#  define _CCCL_HAS_AUTO_CAST_BEFORE_CXX23() 0
#endif // _CCCL_CUDA_COMPILER(NVCC, <, 13)

#if (_CCCL_STD_VER >= 2023 && __cpp_auto_cast >= 202110L) || _CCCL_HAS_AUTO_CAST_BEFORE_CXX23()
#  define _CCCL_AUTO_CAST(expr) auto(expr)
#elif _CCCL_STD_VER < 2020 && _CCCL_COMPILER(MSVC)
#  define _CCCL_AUTO_CAST(expr) (::cuda::std::decay_t<decltype((expr))>) (expr)
#else
#  define _CCCL_AUTO_CAST(expr) static_cast<::cuda::std::decay_t<decltype((expr))>>(expr)
#endif

#endif // _CUDA_STD___UTILITY_AUTO_CAST_H
