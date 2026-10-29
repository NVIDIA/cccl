// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CSTDDEF_TYPES_H
#define _CUDA_STD___CSTDDEF_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED()
#  include <cstddef> // IWYU pragma: export
#else // ^^^ _CCCL_HOSTED() ^^^ / vvv _CCCL_FREESTANDING() vvv
#  if !defined(offsetof)
#    if _CCCL_HAS_BUILTIN(__builtin_offsetof) || _CCCL_COMPILER(MSVC) || _CCCL_COMPILER(GCC)
#      define offsetof(_TYPE, _MEMBER) __builtin_offsetof(_TYPE, _MEMBER)
#    elif _CCCL_COMPILER(NVRTC, >=, 12, 3)
#      define offsetof(_TYPE, _MEMBER) ((::size_t) __INTADDR__(__builtin_addressof(((_TYPE*) 0)->_MEMBER)))
#    elif _CCCL_COMPILER(NVRTC)
#      define offsetof(_TYPE, _MEMBER) ((::size_t) __INTADDR__(&((_TYPE*) 0)->_MEMBER))
#    else // non-constexpr fallback
#      define offsetof(_TYPE, _MEMBER) (::size_t) ((char*) &(((_TYPE*) 0)->_MEMBER) - (char*) 0)
#    endif
#  endif // !offsetof
#endif // _CCCL_FREESTANDING()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_FREESTANDING()
using max_align_t = long double;
#else // ^^^ _CCCL_FREESTANDING() ^^^ / vvv _CCCL_HOSTED() vvv
// Re-use the compiler's <stddef.h> max_align_t where possible.
using ::max_align_t;
#endif // _CCCL_HOSTED()

using nullptr_t = decltype(nullptr);
using ::ptrdiff_t;
using ::size_t;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTDDEF_TYPES_H
