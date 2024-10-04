// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_WAIT_THREADING_SUPPORT_PTHREAD_H
#define _LIBCUDACXX___ATOMIC_WAIT_THREADING_SUPPORT_PTHREAD_H

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

typedef ::timespec __libcpp_timespec_t;

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Mutex
typedef pthread_mutex_t __libcpp_mutex_t;
#  define _LIBCUDACXX_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

typedef pthread_mutex_t __libcpp_recursive_mutex_t;

// Condition Variable
typedef pthread_cond_t __libcpp_condvar_t;
#  define _LIBCUDACXX_CONDVAR_INITIALIZER PTHREAD_COND_INITIALIZER

// Semaphore
#  if defined(__APPLE__)
typedef dispatch_semaphore_t __libcpp_semaphore_t;
#    define _LIBCUDACXX_SEMAPHORE_MAX numeric_limits<long>::max()
#  else // ^^^ __APPLE__ ^^^ / vvv !__APPLE__ vvv
typedef sem_t __libcpp_semaphore_t;
#    define _LIBCUDACXX_SEMAPHORE_MAX SEM_VALUE_MAX
#  endif // !__APPLE__

// Execute once
typedef pthread_once_t __libcpp_exec_once_flag;
#  define _LIBCUDACXX_EXEC_ONCE_INITIALIZER PTHREAD_ONCE_INIT

// Thread id
typedef pthread_t __libcpp_thread_id;

// Thread
#  define _LIBCUDACXX_NULL_THREAD 0U

typedef pthread_t __libcpp_thread_t;

// Thread Local Storage
typedef pthread_key_t __libcpp_tls_key;

#  define _LIBCUDACXX_TLS_DESTRUCTOR_CC

_LIBCUDACXX_HIDE_FROM_ABI __libcpp_timespec_t __libcpp_to_timespec(const _CUDA_VSTD::chrono::nanoseconds& __ns)
{
  using namespace chrono;
  seconds __s = duration_cast<seconds>(__ns);
  __libcpp_timespec_t __ts;
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

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_recursive_mutex_init(__libcpp_recursive_mutex_t* __m)
{
  pthread_mutexattr_t attr;
  int __ec = pthread_mutexattr_init(&attr);
  if (__ec)
  {
    return __ec;
  }
  __ec = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  if (__ec)
  {
    pthread_mutexattr_destroy(&attr);
    return __ec;
  }
  __ec = pthread_mutex_init(__m, &attr);
  if (__ec)
  {
    pthread_mutexattr_destroy(&attr);
    return __ec;
  }
  __ec = pthread_mutexattr_destroy(&attr);
  if (__ec)
  {
    pthread_mutex_destroy(__m);
    return __ec;
  }
  return 0;
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_recursive_mutex_lock(__libcpp_recursive_mutex_t* __m)
{
  return pthread_mutex_lock(__m);
}

_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_recursive_mutex_trylock(__libcpp_recursive_mutex_t* __m)
{
  return pthread_mutex_trylock(__m) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_recursive_mutex_unlock(__libcpp_mutex_t* __m)
{
  return pthread_mutex_unlock(__m);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_recursive_mutex_destroy(__libcpp_recursive_mutex_t* __m)
{
  return pthread_mutex_destroy(__m);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_mutex_lock(__libcpp_mutex_t* __m)
{
  return pthread_mutex_lock(__m);
}

_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_mutex_trylock(__libcpp_mutex_t* __m)
{
  return pthread_mutex_trylock(__m) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_mutex_unlock(__libcpp_mutex_t* __m)
{
  return pthread_mutex_unlock(__m);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_mutex_destroy(__libcpp_mutex_t* __m)
{
  return pthread_mutex_destroy(__m);
}

// Condition Variable
_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_condvar_signal(__libcpp_condvar_t* __cv)
{
  return pthread_cond_signal(__cv);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_condvar_broadcast(__libcpp_condvar_t* __cv)
{
  return pthread_cond_broadcast(__cv);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_condvar_wait(__libcpp_condvar_t* __cv, __libcpp_mutex_t* __m)
{
  return pthread_cond_wait(__cv, __m);
}

_LIBCUDACXX_HIDE_FROM_ABI int
__libcpp_condvar_timedwait(__libcpp_condvar_t* __cv, __libcpp_mutex_t* __m, __libcpp_timespec_t* __ts)
{
  return pthread_cond_timedwait(__cv, __m, __ts);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_condvar_destroy(__libcpp_condvar_t* __cv)
{
  return pthread_cond_destroy(__cv);
}

// Semaphore
#  if defined(__APPLE__)

bool __libcpp_semaphore_init(__libcpp_semaphore_t* __sem, int __init)
{
  return (*__sem = dispatch_semaphore_create(__init)) != nullptr;
}

bool __libcpp_semaphore_destroy(__libcpp_semaphore_t* __sem)
{
  dispatch_release(*__sem);
  return true;
}

bool __libcpp_semaphore_post(__libcpp_semaphore_t* __sem)
{
  dispatch_semaphore_signal(*__sem);
  return true;
}

bool __libcpp_semaphore_wait(__libcpp_semaphore_t* __sem)
{
  return dispatch_semaphore_wait(*__sem, DISPATCH_TIME_FOREVER) == 0;
}

bool __libcpp_semaphore_wait_timed(__libcpp_semaphore_t* __sem, _CUDA_VSTD::chrono::nanoseconds const& __ns)
{
  return dispatch_semaphore_wait(*__sem, dispatch_time(DISPATCH_TIME_NOW, __ns.count())) == 0;
}

#  else // ^^^ __APPLE__ ^^^ / vvv !__APPLE__ vvv

_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_semaphore_init(__libcpp_semaphore_t* __sem, int __init)
{
  return sem_init(__sem, 0, __init) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_semaphore_destroy(__libcpp_semaphore_t* __sem)
{
  return sem_destroy(__sem) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_semaphore_post(__libcpp_semaphore_t* __sem)
{
  return sem_post(__sem) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_semaphore_wait(__libcpp_semaphore_t* __sem)
{
  return sem_wait(__sem) == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI bool
__libcpp_semaphore_wait_timed(__libcpp_semaphore_t* __sem, _CUDA_VSTD::chrono::nanoseconds const& __ns)
{
  __libcpp_timespec_t __ts = __libcpp_to_timespec(__ns);
  return sem_timedwait(__sem, &__ts) == 0;
}

#  endif // !__APPLE__

// Execute once
_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_execute_once(__libcpp_exec_once_flag* flag, void (*init_routine)())
{
  return pthread_once(flag, init_routine);
}

// Thread id
// Returns non-zero if the thread ids are equal, otherwise 0
_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_thread_id_equal(__libcpp_thread_id t1, __libcpp_thread_id t2)
{
  return pthread_equal(t1, t2) != 0;
}

// Returns non-zero if t1 < t2, otherwise 0
_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_thread_id_less(__libcpp_thread_id t1, __libcpp_thread_id t2)
{
  return t1 < t2;
}

// Thread
_LIBCUDACXX_HIDE_FROM_ABI bool __libcpp_thread_isnull(const __libcpp_thread_t* __t)
{
  return *__t == 0;
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_thread_create(__libcpp_thread_t* __t, void* (*__func)(void*), void* __arg)
{
  return pthread_create(__t, 0, __func, __arg);
}

_LIBCUDACXX_HIDE_FROM_ABI __libcpp_thread_id __libcpp_thread_get_current_id()
{
  return pthread_self();
}

_LIBCUDACXX_HIDE_FROM_ABI __libcpp_thread_id __libcpp_thread_get_id(const __libcpp_thread_t* __t)
{
  return *__t;
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_thread_join(__libcpp_thread_t* __t)
{
  return pthread_join(*__t, 0);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_thread_detach(__libcpp_thread_t* __t)
{
  return pthread_detach(*__t);
}

// Thread local storage
_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_tls_create(__libcpp_tls_key* __key, void (*__at_exit)(void*))
{
  return pthread_key_create(__key, __at_exit);
}

_LIBCUDACXX_HIDE_FROM_ABI void* __libcpp_tls_get(__libcpp_tls_key __key)
{
  return pthread_getspecific(__key);
}

_LIBCUDACXX_HIDE_FROM_ABI int __libcpp_tls_set(__libcpp_tls_key __key, void* __p)
{
  return pthread_setspecific(__key, __p);
}

_LIBCUDACXX_HIDE_FROM_ABI void __libcpp_thread_yield()
{
  sched_yield();
}

_LIBCUDACXX_HIDE_FROM_ABI void __libcpp_thread_sleep_for(_CUDA_VSTD::chrono::nanoseconds __ns)
{
  __libcpp_timespec_t __ts = __libcpp_to_timespec(__ns);
  while (nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
    ;
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // !_LIBCUDACXX_HAS_NO_THREADS

#endif // _LIBCUDACXX___ATOMIC_WAIT_THREADING_SUPPORT_PTHREAD_H
