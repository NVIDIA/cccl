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
#include <cuda/std/detail/libcxx/include/cstring>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using ::free;
using ::malloc;

_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __calloc_device(size_t __n, size_t __size) noexcept
{
  void* __ptr{};

  const size_t __nbytes = __n * __size;

  if (::__umul64hi(__n, __size) == 0)
  {
    __ptr = _CUDA_VSTD::malloc(__nbytes);
    if (__ptr != nullptr)
    {
      _CUDA_VSTD::memset(__ptr, 0, __nbytes);
    }
  }

  return __ptr;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* calloc(size_t __n, size_t __size) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return ::std::calloc(__n, __size);), (return _CUDA_VSTD::__calloc_device(__n, __size);))
}

#if !_CCCL_COMPILER(NVRTC)
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST void* __aligned_alloc_host(size_t __nbytes, size_t __align) noexcept
{
  void* __ptr{};
#  if _CCCL_COMPILER(MSVC)
  _CCCL_ASSERT(false, "Use of aligned_alloc in host code is not supported with MSVC");
#  else
  __ptr = ::std::aligned_alloc(__align, __nbytes);
#  endif
  return __ptr;
}
#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __aligned_alloc_device(size_t __nbytes, size_t __align) noexcept
{
  void* __ptr{};
#if _CCCL_CUDA_COMPILER(CLANG)
  _CCCL_ASSERT(false, "Use of aligned_alloc in device code is not supported with clang-cuda");
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG)
  NV_IF_TARGET(NV_IS_DEVICE, (__ptr = ::__nv_aligned_device_malloc(__nbytes, __align);))
#endif // ^^^ !_CCCL_CUDA_COMPILER(CLANG) ^^^
  return __ptr;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* aligned_alloc(size_t __nbytes, size_t __align) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return _CUDA_VSTD::__aligned_alloc_host(__nbytes, __align);),
                    (return _CUDA_VSTD::__aligned_alloc_device(__nbytes, __align);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_MALLOC_H
