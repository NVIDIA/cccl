// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___THREAD_THREADING_SUPPORT_PTHREAD_H
#define _CUDA_STD___THREAD_THREADING_SUPPORT_PTHREAD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_NATIVE_MUTEX(PTHREAD)

#  include <nv/target>

#  include <pthread.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

static_assert(sizeof(__cccl_mutex_t) == sizeof(::pthread_mutex_t), "mutex storage must match pthread_mutex_t");
static_assert(alignof(__cccl_mutex_t) == alignof(::pthread_mutex_t), "mutex storage must match pthread_mutex_t");

_CCCL_HOST_API inline void __cccl_mutex_lock_host(__cccl_mutex_t* __mutex)
{
  ::pthread_mutex_lock(reinterpret_cast<::pthread_mutex_t*>(__mutex));
}

[[nodiscard]] _CCCL_HOST_API inline bool __cccl_mutex_trylock_host(__cccl_mutex_t* __mutex) noexcept
{
  return ::pthread_mutex_trylock(reinterpret_cast<::pthread_mutex_t*>(__mutex)) == 0;
}

_CCCL_HOST_API inline void __cccl_mutex_unlock_host(__cccl_mutex_t* __mutex) noexcept
{
  ::pthread_mutex_unlock(reinterpret_cast<::pthread_mutex_t*>(__mutex));
}

template <thread_scope _Sco>
_CCCL_HOST_DEVICE_API inline void __cccl_mutex_lock(__cccl_mutex_t* __mutex)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (__cccl_mutex_lock_host(__mutex);),
                    (__cccl_mutex_lock_atomic<_Sco, __cuda_atomic_device_backend>(__mutex);))
}

template <thread_scope _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline bool __cccl_mutex_trylock(__cccl_mutex_t* __mutex) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return __cccl_mutex_trylock_host(__mutex);),
                    (return __cccl_mutex_trylock_atomic<_Sco, __cuda_atomic_device_backend>(__mutex);))
}

template <thread_scope _Sco>
_CCCL_HOST_DEVICE_API inline void __cccl_mutex_unlock(__cccl_mutex_t* __mutex) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (__cccl_mutex_unlock_host(__mutex);),
                    (__cccl_mutex_unlock_atomic<_Sco, __cuda_atomic_device_backend>(__mutex);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_NATIVE_MUTEX(PTHREAD)

#if defined(_CCCL_HAS_THREAD_API_PTHREAD) && _CCCL_HOSTED()

#  include <cuda/std/__chrono/duration.h>
#  include <cuda/std/__utility/cmp.h>
#  include <cuda/std/climits>
#  include <cuda/std/ctime>

#  include <errno.h>
#  include <pthread.h>
#  include <sched.h>
#  include <semaphore.h>
#  if defined(__linux__)
#    include <unistd.h>

#    include <linux/futex.h>
#    include <sys/syscall.h>
#  endif // __linux__

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using __cccl_recursive_mutex_t = pthread_mutex_t;

// Condition Variable
using __cccl_condvar_t = pthread_cond_t;
#  define _LIBCUDACXX_CONDVAR_INITIALIZER PTHREAD_COND_INITIALIZER

// Semaphore
using __cccl_semaphore_t = sem_t;
#  define _LIBCUDACXX_SEMAPHORE_MAX SEM_VALUE_MAX

// Execute once
using __cccl_exec_once_flag = pthread_once_t;
#  define _LIBCUDACXX_EXEC_ONCE_INITIALIZER PTHREAD_ONCE_INIT

// Thread id
using __cccl_thread_id = pthread_t;

// Thread
#  define _LIBCUDACXX_NULL_THREAD 0U

using __cccl_thread_t = pthread_t;

// Thread Local Storage
using __cccl_tls_key = pthread_key_t;

#  define _LIBCUDACXX_TLS_DESTRUCTOR_CC

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr timespec __cccl_to_timespec(const ::cuda::std::chrono::nanoseconds& __ns)
{
  constexpr auto __ts_sec_max = numeric_limits<time_t>::max();

  timespec __ts{};
  const auto __s = ::cuda::std::chrono::duration_cast<chrono::seconds>(__ns);

  if (::cuda::std::cmp_less(__s.count(), __ts_sec_max))
  {
    __ts.tv_sec  = static_cast<time_t>(__s.count());
    __ts.tv_nsec = static_cast<decltype(__ts.tv_nsec)>((__ns - __s).count());
  }
  else
  {
    __ts.tv_sec  = __ts_sec_max;
    __ts.tv_nsec = 999'999'999;
  }
  return __ts;
}

// Semaphore

_CCCL_HOST_DEVICE_API inline bool __cccl_semaphore_init(__cccl_semaphore_t* __sem, int __init)
{
  return sem_init(__sem, 0, __init) == 0;
}

_CCCL_HOST_DEVICE_API inline bool __cccl_semaphore_destroy(__cccl_semaphore_t* __sem)
{
  return sem_destroy(__sem) == 0;
}

_CCCL_HOST_DEVICE_API inline bool __cccl_semaphore_post(__cccl_semaphore_t* __sem)
{
  return sem_post(__sem) == 0;
}

_CCCL_HOST_DEVICE_API inline bool __cccl_semaphore_wait(__cccl_semaphore_t* __sem)
{
  return sem_wait(__sem) == 0;
}

_CCCL_HOST_DEVICE_API inline bool
__cccl_semaphore_wait_timed(__cccl_semaphore_t* __sem, ::cuda::std::chrono::nanoseconds const& __ns)
{
  const auto __ts = __cccl_to_timespec(__ns);
  return sem_timedwait(__sem, &__ts) == 0;
}

_CCCL_HOST_DEVICE_API inline void __cccl_thread_yield()
{
  sched_yield();
}

_CCCL_HOST_DEVICE_API inline void __cccl_thread_sleep_for(::cuda::std::chrono::nanoseconds __ns)
{
  auto __ts = __cccl_to_timespec(__ns);
  while (nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
  {
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_HAS_THREAD_API_PTHREAD

#endif // _CUDA_STD___THREAD_THREADING_SUPPORT_PTHREAD_H
