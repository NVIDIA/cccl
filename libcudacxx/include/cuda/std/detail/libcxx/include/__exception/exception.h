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

#ifndef _LIBCUDACXX___EXCEPTION_EXCEPTION_H
#define _LIBCUDACXX___EXCEPTION_EXCEPTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

class _LIBCUDACXX_TYPE_VIS exception
{
public:
  _LIBCUDACXX_INLINE_VISIBILITY exception() noexcept {}
  exception(const exception&) noexcept = default;
  _LIBCUDACXX_INLINE_VISIBILITY virtual ~exception() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual const char* what() const noexcept
  {
    return "cuda::std::exception";
  }
};

class _LIBCUDACXX_TYPE_VIS bad_exception : public exception
{
public:
  _LIBCUDACXX_INLINE_VISIBILITY bad_exception() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual ~bad_exception() noexcept {}
  _LIBCUDACXX_INLINE_VISIBILITY virtual const char* what() const noexcept
  {
    return "cuda::std::bad_exception";
  }
};

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___EXCEPTION_EXCEPTION_H
