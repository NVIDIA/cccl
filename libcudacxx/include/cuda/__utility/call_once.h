//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_CALL_ONCE_H
#define _CUDA___UTILITY_CALL_ONCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__host_stdlib/mutex>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HOSTED()

using __once_flag = ::std::once_flag;

template <class _Fn, class... _Args>
_CCCL_HOST_API void __call_once(__once_flag& __flag, _Fn&& __fn, _Args&&... __args)
{
  ::std::call_once(__flag, ::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...);
}

#else // ^^^ _CCCL_HOSTED() ^^^ / vvv _CCCL_FREESTANDING() vvv

// Needs to be a struct with default value so that `__once_flag flag` correctly initializes the
// state to 0.
struct __once_flag
{
  // uint32_t instead of bool is deliberate:
  //
  // 1. If we ever want to implement a real call_once then we will need 3 states
  //    (uninitialized, in progress, complete).
  // 2. Atomic access to uint32_t is lock-free on the overwhelming majority of platforms.
  ::cuda::std::uint32_t __state_{};
};

template <class _Fn, class... _Args>
_CCCL_HOST_API void __call_once(__once_flag& __flag, _Fn&& __fn, _Args&&... __args)
{
  if (__flag.__state_ == 0)
  {
    ::cuda::std::forward<_Fn>(__fn)(::cuda::std::forward<_Args>(__args)...);
    __flag.__state_ = 1;
  }
}

#endif // ^^^ _CCCL_FREESTANDING() ^^^

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_CALL_ONCE_H
