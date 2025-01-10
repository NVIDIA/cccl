// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_MALLOC_H
#define _LIBCUDACXX___CSTDLIB_MALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using ::free;
using ::malloc;

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* calloc(size_t __n, size_t __size) noexcept
{
  void* __ptr{};

  NV_IF_ELSE_TARGET(
    NV_IS_HOST, (__ptr = ::calloc(__n, __size);), (size_t __nbytes = __n * __size; if (::__umul64hi(__n, __size) == 0) {
      __ptr = ::malloc(__nbytes);
      if (__ptr != nullptr)
      {
        ::memset(__ptr, 0, __nbytes);
      }
    }))

  return __ptr;
}

#if _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER(MSVC)
#  define _LIBCUDACXX_HAS_ALIGNED_ALLOC_HOST 1
#  define _LIBCUDACXX_ALIGNED_ALLOC_HOST     _CCCL_HOST
#else
#  define _LIBCUDACXX_ALIGNED_ALLOC_HOST
#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER(MSVC)

#if _CCCL_HAS_CUDA_COMPILER && !_CCCL_CUDA_COMPILER(CLANG)
#  define _LIBCUDACXX_HAS_ALIGNED_ALLOC_DEVICE 1
#  define _LIBCUDACXX_ALIGNED_ALLOC_DEVICE     _CCCL_DEVICE
#else
#  define _LIBCUDACXX_ALIGNED_ALLOC_DEVICE
#endif // _CCCL_HAS_CUDA_COMPILER && !_CCCL_CUDA_COMPILER(CLANG)

#define _LIBCUDACXX_ALIGNED_ALLOC_EXSPACE _LIBCUDACXX_ALIGNED_ALLOC_HOST _LIBCUDACXX_ALIGNED_ALLOC_DEVICE

_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _LIBCUDACXX_ALIGNED_ALLOC_EXSPACE void*
aligned_alloc(size_t __nbytes, size_t __align) noexcept
{
  NV_IF_TARGET(
    NV_IS_HOST, (return ::aligned_alloc(__align, __nbytes);), (return ::__nv_aligned_device_malloc(__nbytes, __align);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_MALLOC_H
