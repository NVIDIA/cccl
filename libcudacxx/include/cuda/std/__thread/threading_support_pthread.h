// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_PTHREAD_H
#define _LIBCUDACXX___THREAD_THREADING_SUPPORT_PTHREAD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_LIBCUDACXX_HAS_NO_THREADS) && defined(_LIBCUDACXX_HAS_THREAD_API_PTHREAD)

#  include <cuda/std/chrono>
#  include <cuda/std/climits>

#  include <errno.h>
#  include <pthread.h>
#  include <sched.h>
#  include <semaphore.h>
#  if defined(__APPLE__)
#    include <dispatch/dispatch.h>
#  endif // __APPLE__
#  if defined(__linux__)
#    include <linux/futex.h>
#    include <sys/syscall.h>
#    include <unistd.h>
#  endif // __linux__

_CCCL_PUSH_MACROS

typedef ::timespec __cccl_timespec_t;

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Mutex
typedef pthread_mutex_t __cccl_mutex_t;
#  define _LIBCUDACXX_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

typedef pthread_mutex_t __cccl_recursive_mutex_t;

// Condition Variable
typedef pthread_cond_t __cccl_condvar_t;
#  define _LIBCUDACXX_CONDVAR_INITIALIZER PTHREAD_COND_INITIALIZER

// Semaphore
#  if defined(__APPLE__)
typedef dispatch_semaphore_t __cccl_semaphore_t;
#    define _LIBCUDACXX_SEMAPHORE_MAX numeric_limits<long>::max()
#  else // ^^^ __APPLE__ ^^^ / vvv !__APPLE__ vvv
typedef sem_t __cccl_semaphore_t;
#    define _LIBCUDACXX_SEMAPHORE_MAX SEM_VALUE_MAX
#  endif // !__APPLE__

// Execute once
typedef pthread_once_t __cccl_exec_once_flag;
#  define _LIBCUDACXX_EXEC_ONCE_INITIALIZER PTHREAD_ONCE_INIT

// Thread id
typedef pthread_t __cccl_thread_id;

// Thread
#  define _LIBCUDACXX_NULL_THREAD 0U

typedef pthread_t __cccl_thread_t;

// Thread Local Storage
typedef pthread_key_t __cccl_tls_key;

#  define _LIBCUDACXX_TLS_DESTRUCTOR_CC

_LIBCUDACXX_HIDE_FROM_ABI __cccl_timespec_t __cccl_to_timespec(const _CUDA_VSTD::chrono::nanoseconds& __ns)
{
  using namespace chrono;
  seconds __s = duration_cast<seconds>(__ns);
  __cccl_timespec_t __ts;
  typedef decltype(__ts.tv_sec) ts_sec;
  constexpr ts_sec __ts_sec_max = numeric_limits<ts_sec>::max();

  if (__s.count() < __ts_sec_max)
  {
    __ts.tv_sec  = static_cast<ts_sec>(__s.count());
    __ts.tv_nsec = static_cast<decltype(__ts.tv_nsec)>((__ns - __s).count());
  }
  else
  {
    __ts.tv_sec  = __ts_sec_max;
    __ts.tv_nsec = 999999999; // (10^9 - 1)
  }
  return __ts;
}

// Semaphore
#  if defined(__APPLE__)

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_init(__cccl_semaphore_t* __sem, int __init)
{
  return (*__sem = dispatch_semaphore_create(__init)) != nullptr;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_destroy(__cccl_semaphore_t* __sem)
{
  dispatch_release(*__sem);
  return true;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_post(__cccl_semaphore_t* __sem)
{
  dispatch_semaphore_signal(*__sem);
  return true;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_wait(__cccl_semaphore_t* __sem)
{
  return dispatch_semaphore_wait(*__sem, DISPATCH_TIME_FOREVER) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool
__cccl_semaphore_wait_timed(__cccl_semaphore_t* __sem, _CUDA_VSTD::chrono::nanoseconds const& __ns)
{
  return dispatch_semaphore_wait(*__sem, dispatch_time(DISPATCH_TIME_NOW, __ns.count())) == 0;
}

#  else // ^^^ __APPLE__ ^^^ / vvv !__APPLE__ vvv

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_init(__cccl_semaphore_t* __sem, int __init)
{
  return sem_init(__sem, 0, __init) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_destroy(__cccl_semaphore_t* __sem)
{
  return sem_destroy(__sem) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_post(__cccl_semaphore_t* __sem)
{
  return sem_post(__sem) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __cccl_semaphore_wait(__cccl_semaphore_t* __sem)
{
  return sem_wait(__sem) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool
__cccl_semaphore_wait_timed(__cccl_semaphore_t* __sem, _CUDA_VSTD::chrono::nanoseconds const& __ns)
{
  __cccl_timespec_t __ts = __cccl_to_timespec(__ns);
  return sem_timedwait(__sem, &__ts) == 0;
}

#  endif // !__APPLE__

_LIBCUDACXX_HIDE_FROM_ABI void __cccl_thread_yield()
{
  sched_yield();
}

_LIBCUDACXX_HIDE_FROM_ABI void __cccl_thread_sleep_for(_CUDA_VSTD::chrono::nanoseconds __ns)
{
  __cccl_timespec_t __ts = __cccl_to_timespec(__ns);
  while (nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
    ;
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // !_LIBCUDACXX_HAS_NO_THREADS

#endif // _LIBCUDACXX___THREAD_THREADING_SUPPORT_PTHREAD_H
