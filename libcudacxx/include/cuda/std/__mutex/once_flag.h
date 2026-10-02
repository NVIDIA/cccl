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

#include <cuda/std/__atomic/functions/dispatch.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/__utility/forward.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Unscoped so the enumerators convert to the unsigned state word the atomics load and store.
enum __once_flag_state : unsigned int // NOLINT(cppcoreguidelines-use-enum-class)
{
  __unset    = 0,
  __pending  = 1,
  __complete = ~0u,
};

template <thread_scope _Sco = thread_scope::thread_scope_system>
struct once_flag
{
  _CCCL_HIDE_FROM_ABI constexpr once_flag() noexcept = default;

  once_flag(const once_flag&)            = delete;
  once_flag& operator=(const once_flag&) = delete;

  static constexpr thread_scope _Scope = _Sco;

private:
  __once_flag_state __state_ = __once_flag_state::__unset;

  // No exception specification here: nvcc's front end crashes in gen_exception_specification
  // when a friend function template inside this class carries a dependent noexcept.
  template <thread_scope _Scope, class _Backend, class _Fn, class... _Args>
  _CCCL_HOST_DEVICE_API friend void __call_once(once_flag<_Scope>&, _Fn&&, _Args&&...);
};

// Rolls the flag back to unset when the callable throws, so another thread can attempt the call.
template <thread_scope _Sco, class _Backend>
struct __once_flag_clear
{
  __once_flag_state* __flag;

  _CCCL_HOST_DEVICE_API void operator()() const noexcept
  {
    ::cuda::std::__cuda_atomic_store_dispatch(
      _Backend{}, __flag, __once_flag_state::__unset, memory_order_relaxed, ::cuda::std::__scope_to_tag<_Sco>{});
  }
};

// One thread claims the flag. The others spin until that call finishes or, if it throws, until the flag is cleared.
template <thread_scope _Sco, class _Backend, class _Fn, class... _Args>
_CCCL_HOST_DEVICE_API void __call_once(once_flag<_Sco>& __flag, _Fn&& __fn, _Args&&... __args)
{
  __once_flag_state* __state = &__flag.__state_;
  constexpr _Backend __backend{};
  constexpr ::cuda::std::__scope_to_tag<_Sco> __scope{};
  for (;;)
  {
    auto __current = ::cuda::std::__cuda_atomic_load_dispatch(__backend, __state, memory_order_acquire, __scope);
    while (__current == __once_flag_state::__pending)
    {
      __current = ::cuda::std::__cuda_atomic_load_dispatch(__backend, __state, memory_order_acquire, __scope);
    }
    if (__current == __once_flag_state::__complete)
    {
      return;
    }

    __once_flag_state __expected = __once_flag_state::__unset;
    if (::cuda::std::__cuda_atomic_compare_exchange_dispatch(
          __backend,
          __state,
          &__expected,
          __pending,
          ::cuda::std::__cuda_atomic_cas_strong{},
          memory_order_acq_rel,
          memory_order_acquire,
          __scope))
    {
      auto __guard = ::cuda::std::__make_exception_guard(__once_flag_clear<_Sco, _Backend>{__state});
      ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...);
      ::cuda::std::__cuda_atomic_store_dispatch(__backend, __state, __complete, memory_order_release, __scope);
      __guard.__complete();
      return;
    }
  }
}

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using once_flag = ::cuda::once_flag<thread_scope::thread_scope_system>;

template <thread_scope _Sco, class _Fn, class... _Args>
_CCCL_HOST_DEVICE_API void call_once(::cuda::once_flag<_Sco>& __flag, _Fn&& __fn, _Args&&... __args) noexcept(
  is_nothrow_invocable_v<_Fn, _Args...>)
{
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (::cuda::__call_once<_Sco, __cuda_atomic_host_backend>(
       __flag, ::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...);),
    (::cuda::__call_once<_Sco, __cuda_atomic_device_backend>(
       __flag, ::cuda::std::forward<_Fn>(__fn), ::cuda::std::forward<_Args>(__args)...);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MUTEX_ONCE_FLAG_H
