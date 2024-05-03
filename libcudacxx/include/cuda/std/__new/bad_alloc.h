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

#ifndef _LIBCUDACXX___NEW_BAD_ALLOC_H
#define _LIBCUDACXX___NEW_BAD_ALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/exception.h>
#include <cuda/std/__exception/terminate.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

class _LIBCUDACXX_TYPE_VIS bad_alloc : public exception
{
public:
  _LIBCUDACXX_INLINE_VISIBILITY bad_alloc() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual ~bad_alloc() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual const char* what() const noexcept
  {
    return "cuda::std::bad_alloc";
  }
};

class _LIBCUDACXX_TYPE_VIS bad_array_new_length : public bad_alloc
{
public:
  _LIBCUDACXX_INLINE_VISIBILITY bad_array_new_length() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual ~bad_array_new_length() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual const char* what() const noexcept
  {
    return "cuda::std::bad_array_new_length";
  }
};

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NORETURN inline _LIBCUDACXX_INLINE_VISIBILITY void __throw_bad_alloc()
{
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw _CUDA_VSTD_NOVERSION::bad_alloc();), (_CUDA_VSTD_NOVERSION::terminate();))
#else
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_LIBCUDACXX_NO_EXCEPTIONS
}

_CCCL_NORETURN inline _LIBCUDACXX_INLINE_VISIBILITY void __throw_bad_array_new_length()
{
#ifndef _LIBCUDACXX_NO_EXCEPTIONS
  NV_IF_ELSE_TARGET(
    NV_IS_HOST, (throw _CUDA_VSTD_NOVERSION::bad_array_new_length();), (_CUDA_VSTD_NOVERSION::terminate();))
#else
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_LIBCUDACXX_NO_EXCEPTIONS
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NEW_BAD_ALLOC_H
