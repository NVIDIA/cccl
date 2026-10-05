//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MUTEX_ONCE_FLAG_H
#define _CUDA_STD___MUTEX_ONCE_FLAG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__thread/threading_support.h>
#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Unlocks the once_flag mutex if the callable throws, so another caller can try again.
template <thread_scope _Sco>
struct __once_flag_unlock
{
  ::cuda::std::__cccl_mutex_t* __mutex;

  _CCCL_HOST_DEVICE_API void operator()() const noexcept
  {
    ::cuda::std::__cccl_mutex_unlock<_Sco>(__mutex);
  }
};

template <thread_scope _Sco = thread_scope::thread_scope_system>
struct once_flag
{
  _CCCL_HIDE_FROM_ABI constexpr once_flag() noexcept = default;

  once_flag(const once_flag&)            = delete;
  once_flag& operator=(const once_flag&) = delete;

  static constexpr thread_scope _Scope = _Sco;

private:
  bool __complete_                   = false;
  ::cuda::std::__cccl_mutex_t __mut_{};

  // No exception specification here: nvcc's front end crashes in gen_exception_specification
  // when a friend function template inside this class carries a dependent noexcept.
  template <thread_scope _Scope, class _Fn, class... _Args>
  _CCCL_HOST_DEVICE_API friend void __call_once(once_flag<_Scope>&, _Fn&&, _Args&&...);
};

// The mutex is the host lock when one exists, and the ticket lock otherwise.
template <thread_scope _Sco, class _Fn, class... _Args>
_CCCL_HOST_DEVICE_API void __call_once(once_flag<_Sco>& __flag, _Fn&& __fn, _Args&&... __args)
{
  ::cuda::std::__cccl_mutex_lock<_Sco>(&__flag.__mut_);
  auto __guard = ::cuda::std::__make_exception_guard(__once_flag_unlock<_Sco>{&__flag.__mut_});
  if (!__flag.__complete_)
  {
    ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...);
    __flag.__complete_ = true;
  }
  __guard.__complete();
  ::cuda::std::__cccl_mutex_unlock<_Sco>(&__flag.__mut_);
}

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using once_flag = ::cuda::once_flag<thread_scope::thread_scope_system>;

template <thread_scope _Sco, class _Fn, class... _Args>
_CCCL_HOST_DEVICE_API void call_once(::cuda::once_flag<_Sco>& __flag, _Fn&& __fn, _Args&&... __args) noexcept(
  is_nothrow_invocable_v<_Fn, _Args...>)
{
  ::cuda::__call_once(__flag, ::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MUTEX_ONCE_FLAG_H
