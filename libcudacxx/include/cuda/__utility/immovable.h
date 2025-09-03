//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_IMMOVABLE_H
#define _CUDA___UTILITY_IMMOVABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// BUG (gcc#98995): copy elision fails when initializing a [[no_unique_address]] field
// from a function returning an object of class type by value.
// See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=98995
#if _CCCL_COMPILER(GCC)
// By declaring the move constructor but not defining it, any TU that ODR-uses the move
// constructor will cause a linker error.
#  define _CCCL_IMMOVABLE(_XP) _CCCL_API _XP(_XP&&) noexcept
#else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
#  define _CCCL_IMMOVABLE(_XP) _XP(_XP&&) = delete
#endif // !_CCCL_COMPILER(GCC)

// Classes can inherit from this type to become immovable.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __immovable
{
  _CCCL_HIDE_FROM_ABI __immovable() = default;
  _CCCL_IMMOVABLE(__immovable);
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_IMMOVABLE_H
