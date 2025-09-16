//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CSTRING_MEMCPY
#define _CUDA_STD___CSTRING_MEMCPY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/check_address.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstring>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using ::size_t;

// We cannot redefine memcpy before CUDA 13.2 because of CUDA Math API nvbug 5452563.
// More in detail, some CUDA Math API functions call 'memcpy' without namespace qualification, causing name ambiguity.
#if _CCCL_CTK_AT_LEAST(13, 2)

_CCCL_API inline void* memcpy(void* __dest, const void* __src, size_t __count) noexcept
{
  _CCCL_ASSERT(::cuda::__is_valid_address_range(__src, __count), "memcpy: source range is invalid");
  _CCCL_ASSERT(::cuda::__is_valid_address_range(__dest, __count), "memcpy: destination range is invalid");
  _CCCL_ASSERT(::cuda::__are_ptrs_overlapping(__src, __dest, __count), "memcpy: source and destination overlap");
  return ::memcpy(__dest, __src, __count);
}

#else // < CUDA 13.2

using ::memcpy;

#endif // _CCCL_CTK_AT_LEAST(13, 2)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CSTRING_MEMCPY
