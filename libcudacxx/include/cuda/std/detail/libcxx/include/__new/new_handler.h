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

#ifndef _LIBCUDACXX___NEW_NEW_HANDLER_H
#define _LIBCUDACXX___NEW_NEW_HANDLER_H

#ifndef __cuda_std__
#  include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION // purposefully not using versioning namespace

typedef void (*new_handler)();

#ifdef __CUDA_ARCH__
__device__
#endif // __CUDA_ARCH__
  static _LIBCUDACXX_SAFE_STATIC _CUDA_VSTD::atomic<new_handler>
    __cccl_new_handler{nullptr};

inline _LIBCUDACXX_INLINE_VISIBILITY new_handler set_new_handler(new_handler __func) noexcept
{
  return _CUDA_VSTD::atomic_exchange(&__cccl_new_handler, __func);
}
inline _LIBCUDACXX_INLINE_VISIBILITY new_handler get_new_handler() noexcept
{
  return _CUDA_VSTD::atomic_load(&__cccl_new_handler);
}

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#endif // _LIBCUDACXX___NEW_NEW_HANDLER_H
