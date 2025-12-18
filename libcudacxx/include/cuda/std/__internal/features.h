//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_FEATURES_H
#define _CUDA_STD___INTERNAL_FEATURES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS() (!_CCCL_COMPILER(CLANG) || _CCCL_STD_VER >= 2020)
#define _LIBCUDACXX_HAS_MONOTONIC_CLOCK()       0
#define _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()    0

#if _CCCL_CUDA_COMPILATION() || __cpp_aligned_new < 201606
#  define _LIBCUDACXX_HAS_ALIGNED_ALLOCATION() 0
#else
#  define _LIBCUDACXX_HAS_ALIGNED_ALLOCATION() 1
#endif // !_CCCL_CUDA_COMPILATION() && __cpp_aligned_new >= 201606

// We need `is_constant_evaluated` for clang and gcc. MSVC also needs extensive rework
#if !defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif _CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif _CCCL_COMPILER(MSVC)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif _CCCL_CUDA_COMPILER(CLANG)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#else
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 1
#endif

#if _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS()
#  define _CCCL_CONSTEXPR_COMPLEX constexpr
#else
#  define _CCCL_CONSTEXPR_COMPLEX
#endif // !_LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS()

#ifndef _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#  define _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#endif // _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES

// libcu++ requires host device support for its tests. Until then restrict usage to at least 12.2
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
#  define _LIBCUDACXX_HAS_NVFP16() 1
#else
#  define _LIBCUDACXX_HAS_NVFP16() 0
#endif // _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)

// libcu++ requires host device support for its tests. Until then restrict usage to at least 12.2
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
#  define _LIBCUDACXX_HAS_NVBF16() 1
#else
#  define _LIBCUDACXX_HAS_NVBF16() 0
#endif // _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)

#if _CCCL_COMPILER(MSVC)
#  define _CCCL_ALIGNAS_TYPE(x) alignas(x)
#  define _CCCL_ALIGNAS(x)      __declspec(align(x))
#elif _CCCL_HAS_FEATURE(cxx_alignas)
#  define _CCCL_ALIGNAS_TYPE(x) alignas(x)
#  define _CCCL_ALIGNAS(x)      alignas(x)
#else
#  define _CCCL_ALIGNAS_TYPE(x) __attribute__((__aligned__(alignof(x))))
#  define _CCCL_ALIGNAS(x)      __attribute__((__aligned__(x)))
#endif // !_CCCL_COMPILER(MSVC) && !_CCCL_HAS_FEATURE(cxx_alignas)

// We can only expose constexpr allocations if the compiler supports it
// For now disable constexpr allocation support until we can actually use
#if 0 && defined(__cpp_constexpr_dynamic_alloc) && defined(__cpp_lib_constexpr_dynamic_alloc) && _CCCL_STD_VER >= 2020 \
  && !_CCCL_COMPILER(NVRTC)
#  define _CCCL_HAS_CONSTEXPR_ALLOCATION
#  define _CCCL_CONSTEXPR_CXX20_ALLOCATION constexpr
#else // ^^^ __cpp_constexpr_dynamic_alloc ^^^ / vvv !__cpp_constexpr_dynamic_alloc vvv
#  define _CCCL_CONSTEXPR_CXX20_ALLOCATION
#endif

// Enable removed C++17 features
#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_FEATURES)
#  define _LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS
#endif // _LIBCUDACXX_ENABLE_CXX17_REMOVED_FEATURES

#ifndef _CCCL_DISABLE_ADDITIONAL_DIAGNOSTICS
#  define _CCCL_DIAGNOSE_WARNING(_COND, _MSG) _CCCL_DIAGNOSE_IF(_COND, _MSG, "warning")
#  define _CCCL_DIAGNOSE_ERROR(_COND, _MSG)   _CCCL_DIAGNOSE_IF(_COND, _MSG, "error")
#else
#  define _CCCL_DIAGNOSE_WARNING(_COND, _MSG)
#  define _CCCL_DIAGNOSE_ERROR(_COND, _MSG)
#endif

#endif // _CUDA_STD___INTERNAL_FEATURES_H
