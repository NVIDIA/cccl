//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_ATOMIC_H
#define _CUDA_STD___INTERNAL_ATOMIC_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__internal/features.h>

#if _CCCL_CUDA_COMPILATION()
#  define _CCCL_ATOMIC_ALWAYS_LOCK_FREE(size, ptr) (size <= 8)
#elif _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(GCC)
#  define _CCCL_ATOMIC_ALWAYS_LOCK_FREE(...) __atomic_always_lock_free(__VA_ARGS__)
#endif // _CCCL_CUDA_COMPILER

// Enable bypassing automatic storage checks in atomics when using CTK 12.2 and below and if NDEBUG is defined.
// A compiler bug prevents the safe use of `__is_local` and PTX spacep until after 13.0.
#ifndef _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#  if _CCCL_CUDACC_BELOW(13, 1) && !defined(NDEBUG)
#    define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE
#  endif // _CCCL_CUDACC_BELOW(13, 1)
#endif // _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE

#define _CCCL_ATOMIC_FLAG_TYPE int

// Clang provides 128b atomics as a builtin
#if defined(CCCL_ENABLE_EXPERIMENTAL_HOST_ATOMICS_128B)
#  define _CCCL_HOST_128_ATOMICS_ENABLED() 1
#  define _CCCL_HOST_128_ATOMICS_MAYBE()   0
// GCC does not provide 128b atomics, but they may be available as a library, this requires opt-in usage.
// See: https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html "-mcx16" for more
#elif _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(GCC)
#  define _CCCL_HOST_128_ATOMICS_ENABLED() 0
#  define _CCCL_HOST_128_ATOMICS_MAYBE()   1
#else
#  define _CCCL_HOST_128_ATOMICS_ENABLED() 0
#  define _CCCL_HOST_128_ATOMICS_MAYBE()   0
#endif

#endif // _CUDA_STD___INTERNAL_ATOMIC_H
