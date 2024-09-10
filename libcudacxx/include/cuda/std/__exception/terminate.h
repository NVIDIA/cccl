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

#ifndef _LIBCUDACXX___EXCEPTION_TERMINATE_H
#define _LIBCUDACXX___EXCEPTION_TERMINATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdlib> // ::exit

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void __cccl_terminate() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (::exit(-1);), (__trap();))
  _CCCL_UNREACHABLE();
}

#if 0 // Expose once atomic is universally available

typedef void (*terminate_handler)();

#  ifdef __CUDA_ARCH__
__device__
#  endif // __CUDA_ARCH__
  static _LIBCUDACXX_SAFE_STATIC _CUDA_VSTD::atomic<terminate_handler>
    __cccl_terminate_handler{&__cccl_terminate};

_LIBCUDACXX_HIDE_FROM_ABI  terminate_handler set_terminate(terminate_handler __func) noexcept
{
  return __cccl_terminate_handler.exchange(__func);
}
_LIBCUDACXX_HIDE_FROM_ABI  terminate_handler get_terminate() noexcept
{
  return __cccl_terminate_handler.load(__func);
}

#endif

_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void terminate() noexcept
{
  __cccl_terminate();
  _CCCL_UNREACHABLE();
}

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

_CCCL_DIAG_POP

#endif // _LIBCUDACXX___EXCEPTION_TERMINATE_H
