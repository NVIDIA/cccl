// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_WIN32_H
#define _LIBCUDACXX___THREAD_THREADING_SUPPORT_WIN32_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_LIBCUDACXX_HAS_NO_THREADS) && defined(_LIBCUDACXX_HAS_THREAD_API_WIN32)

#  include <cuda/std/chrono>

#  include <process.h>
#  include <windows.h>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Mutex
typedef void* __cccl_mutex_t;
#  define _LIBCUDACXX_MUTEX_INITIALIZER 0

#  if defined(_M_IX86) || defined(__i386__) || defined(_M_ARM) || defined(__arm__)
typedef void* __cccl_recursive_mutex_t[6];
#  elif defined(_M_AMD64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__)
typedef void* __cccl_recursive_mutex_t[5];
#  else
#    error Unsupported architecture
#  endif

// Condition Variable
typedef void* __cccl_condvar_t;
#  define _LIBCUDACXX_CONDVAR_INITIALIZER 0

// Semaphore
typedef void* __cccl_semaphore_t;

// Execute Once
typedef void* __cccl_exec_once_flag;
#  define _LIBCUDACXX_EXEC_ONCE_INITIALIZER 0

// Thread ID
typedef long __cccl_thread_id;

// Thread
#  define _LIBCUDACXX_NULL_THREAD 0U

typedef void* __cccl_thread_t;

// Thread Local Storage
typedef long __cccl_tls_key;

#  define _LIBCUDACXX_TLS_DESTRUCTOR_CC __stdcall

_LIBCUDACXX_HIDE_FROM_ABI void __cccl_thread_yield()
{
  SwitchToThread();
}

_LIBCUDACXX_HIDE_FROM_ABI void __cccl_thread_sleep_for(chrono::nanoseconds __ns)
{
  using namespace chrono;
  // round-up to the nearest millisecond
  milliseconds __ms = duration_cast<milliseconds>(__ns + chrono::nanoseconds(999999));
  Sleep(static_cast<DWORD>(__ms.count()));
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // !_LIBCUDACXX_HAS_NO_THREADS && _LIBCUDACXX_HAS_THREAD_API_WIN32

#endif // _LIBCUDACXX___THREAD_THREADING_SUPPORT_H
