//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___THREAD_THREADING_SUPPORT_FALLBACK_H
#define _CUDA_STD___THREAD_THREADING_SUPPORT_FALLBACK_H

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

#include <nv/target>

// Storage for `__cccl_mutex_t`. The first eight bytes are the ticket lock. When a platform lock exists, the object is
// the same size and alignment as that lock so the host path can call it in place. Zero-initialization is an unlocked
// glibc/musl pthread mutex and SRWLOCK_INIT.
#if defined(_CCCL_HAS_THREAD_API_PTHREAD)
// glibc LP64 sizes from `sysdeps/{x86,aarch64}/nptl/bits/pthreadtypes-arch.h`:
// `__SIZEOF_PTHREAD_MUTEX_T` is 40 on x86_64 and 48 on aarch64. CCCL hosts are those two.
// `_CCCL_OS(LINUX)` also matches Apple, Android, QNX, and NVRTC, which do not use these sizes.
#  if _CCCL_OS(LINUX) && !_CCCL_OS(APPLE) && !_CCCL_OS(ANDROID) && !_CCCL_OS(QNX) && _CCCL_HOST_ARCH(X86_64)
#    define _CCCL_MUTEX_ABI_BYTES 40
#  elif _CCCL_OS(LINUX) && !_CCCL_OS(APPLE) && !_CCCL_OS(ANDROID) && !_CCCL_OS(QNX) && _CCCL_HOST_ARCH(ARM64)
#    define _CCCL_MUTEX_ABI_BYTES 48
#  endif
#elif defined(_CCCL_HAS_THREAD_API_WIN32)
#  define _CCCL_MUTEX_ABI_BYTES 8
#else // ^^^ no host mutex ABI ^^^ / vvv storage is the ticket lock vvv
#  define _CCCL_MUTEX_ABI_BYTES 8
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <unsigned int _Bytes>
struct __cccl_mutex_storage
{
  static_assert(_Bytes >= 8, "platform mutex is smaller than the ticket lock");
  unsigned int __serving;
  unsigned int __next;
  unsigned char __opaque[_Bytes - 8];
};

template <>
struct __cccl_mutex_storage<8>
{
  unsigned int __serving;
  unsigned int __next;
};

struct alignas(8) __cccl_mutex_t : __cccl_mutex_storage<_CCCL_MUTEX_ABI_BYTES>
{};

// Ticket lock. Each acquirer takes the next ticket and waits until it is served.
// try_lock does not take a ticket unless the mutex is free, so it cannot barge ahead of a waiter.
template <thread_scope _Sco, class _Backend>
_CCCL_HOST_DEVICE_API void __cccl_mutex_lock_atomic(__cccl_mutex_t* __mutex)
{
  constexpr _Backend __backend{};
  constexpr ::cuda::std::__scope_to_tag<_Sco> __scope{};
  const unsigned int __ticket =
    ::cuda::std::__cuda_atomic_fetch_add_dispatch(__backend, &__mutex->__next, 1u, memory_order_relaxed, __scope);
  while (::cuda::std::__cuda_atomic_load_dispatch(__backend, &__mutex->__serving, memory_order_acquire, __scope)
         != __ticket)
  {
  }
}

template <thread_scope _Sco, class _Backend>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __cccl_mutex_trylock_atomic(__cccl_mutex_t* __mutex) noexcept
{
  constexpr _Backend __backend{};
  constexpr ::cuda::std::__scope_to_tag<_Sco> __scope{};
  unsigned int __expected =
    ::cuda::std::__cuda_atomic_load_dispatch(__backend, &__mutex->__next, memory_order_relaxed, __scope);
  const unsigned int __serving_now =
    ::cuda::std::__cuda_atomic_load_dispatch(__backend, &__mutex->__serving, memory_order_acquire, __scope);
  if (__expected != __serving_now)
  {
    return false;
  }
  return ::cuda::std::__cuda_atomic_compare_exchange_dispatch(
    __backend,
    &__mutex->__next,
    &__expected,
    __expected + 1u,
    __cuda_atomic_cas_strong{},
    memory_order_acq_rel,
    memory_order_acquire,
    __scope);
}

template <thread_scope _Sco, class _Backend>
_CCCL_HOST_DEVICE_API void __cccl_mutex_unlock_atomic(__cccl_mutex_t* __mutex) noexcept
{
  constexpr _Backend __backend{};
  constexpr ::cuda::std::__scope_to_tag<_Sco> __scope{};
  (void) ::cuda::std::__cuda_atomic_fetch_add_dispatch(
    __backend, &__mutex->__serving, 1u, memory_order_release, __scope);
}

// No native mutex: the public entry points are the ticket lock. A pthread or SRWLOCK build defines these in the
// platform header and calls `__cccl_mutex_lock_atomic` only on device.
#if !_CCCL_HAS_NATIVE_MUTEX()

template <thread_scope _Sco>
_CCCL_HOST_DEVICE_API inline void __cccl_mutex_lock(__cccl_mutex_t* __mutex)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (__cccl_mutex_lock_atomic<_Sco, __cuda_atomic_host_backend>(__mutex);),
                    (__cccl_mutex_lock_atomic<_Sco, __cuda_atomic_device_backend>(__mutex);))
}

template <thread_scope _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline bool __cccl_mutex_trylock(__cccl_mutex_t* __mutex) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return __cccl_mutex_trylock_atomic<_Sco, __cuda_atomic_host_backend>(__mutex);),
                    (return __cccl_mutex_trylock_atomic<_Sco, __cuda_atomic_device_backend>(__mutex);))
}

template <thread_scope _Sco>
_CCCL_HOST_DEVICE_API inline void __cccl_mutex_unlock(__cccl_mutex_t* __mutex) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (__cccl_mutex_unlock_atomic<_Sco, __cuda_atomic_host_backend>(__mutex);),
                    (__cccl_mutex_unlock_atomic<_Sco, __cuda_atomic_device_backend>(__mutex);))
}

#endif // !_CCCL_HAS_NATIVE_MUTEX()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___THREAD_THREADING_SUPPORT_FALLBACK_H
