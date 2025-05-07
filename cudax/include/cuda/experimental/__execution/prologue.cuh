//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// IMPORTANT: This file intionally lacks a header guard.

#include <cuda/std/detail/__config>

#if defined(_CUDAX_ASYNC_PROLOGUE_INCLUDED)
#  error multiple inclusion of prologue.cuh
#endif

#define _CUDAX_ASYNC_PROLOGUE_INCLUDED

_CCCL_PUSH_MACROS
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif
#ifdef interface
#  undef interface
#endif

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wsubobject-linkage")
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-value")
_CCCL_DIAG_SUPPRESS_MSVC(4848) // [[no_unique_address]] prior to C++20 as a vendor extension

_CCCL_DIAG_SUPPRESS_GCC("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_CLANG("-Wmissing-braces")
_CCCL_DIAG_SUPPRESS_MSVC(5246) // missing braces around initializer

// BUG (gcc#98995): copy elision fails when initializing a [[no_unique_address]] field
// from a function returning an object of class type by value.
// See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98995
#if _CCCL_COMPILER(GCC)
// By declaring the move constructor but not defining it, any TU that ODR-uses the move
// constructor will cause a linker error.
#  define _CCCL_IMMOVABLE_OPSTATE(_XP) _CCCL_API _XP(_XP&&) noexcept
#else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
#  define _CCCL_IMMOVABLE_OPSTATE(_XP) _XP(_XP&&) = delete
#endif // !_CCCL_COMPILER(GCC)
