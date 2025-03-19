// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CSTDLIB_ALIGNED_ALLOC_H
#define _LIBCUDACXX___CSTDLIB_ALIGNED_ALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__cstdlib/malloc.h>
#include <cuda/std/cstring>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

#if _CCCL_HAS_CUDA_COMPILER()
extern "C" _CCCL_DEVICE void* __cuda_syscall_aligned_malloc(size_t, size_t);
#endif // _CCCL_HAS_CUDA_COMPILER()

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if !_CCCL_COMPILER(NVRTC)
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST void* __aligned_alloc_host(size_t __nbytes, size_t __align) noexcept
{
#  if _CCCL_COMPILER(MSVC)
  _LIBCUDACXX_UNUSED_VAR(__nbytes);
  _LIBCUDACXX_UNUSED_VAR(__align);
  _CCCL_ASSERT(false, "Use of aligned_alloc in host code is not supported with MSVC");
  return nullptr;
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  return ::aligned_alloc(__align, __nbytes);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* aligned_alloc(size_t __nbytes, size_t __align) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return _CUDA_VSTD::__aligned_alloc_host(__nbytes, __align);),
                    (return ::__cuda_syscall_aligned_malloc(__nbytes, __align);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_ALIGNED_ALLOC_H
