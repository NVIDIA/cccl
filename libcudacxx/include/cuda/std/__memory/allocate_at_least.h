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

#ifndef _CUDA_STD___MEMORY_ALLOCATE_AT_LEAST_H
#define _CUDA_STD___MEMORY_ALLOCATE_AT_LEAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_STD_VER >= 2023
template <class _Pointer>
struct allocation_result
{
  _Pointer ptr;
  size_t count;
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(allocation_result);

template <class _Alloc>
[[nodiscard]] _CCCL_API constexpr allocation_result<typename allocator_traits<_Alloc>::pointer>
allocate_at_least(_Alloc& __alloc, size_t __n)
{
  if constexpr (requires { __alloc.allocate_at_least(__n); })
  {
    return __alloc.allocate_at_least(__n);
  }
  else
  {
    return {__alloc.allocate(__n), __n};
  }
}

template <class _Alloc>
[[nodiscard]] _CCCL_API constexpr auto __allocate_at_least(_Alloc& __alloc, size_t __n)
{
  return ::cuda::std::allocate_at_least(__alloc, __n);
}
#else // ^^^ _CCCL_STD_VER >= 2023 ^^^ / vvv _CCCL_STD_VER < 2023 vvv
template <class _Pointer>
struct __allocation_result
{
  _Pointer ptr;
  size_t count;
};

template <class _Alloc>
[[nodiscard]] _CCCL_API constexpr __allocation_result<typename allocator_traits<_Alloc>::pointer>
__allocate_at_least(_Alloc& __alloc, size_t __n)
{
  return {__alloc.allocate(__n), __n};
}

#endif // _CCCL_STD_VER >= 2023

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_ALLOCATE_AT_LEAST_H
