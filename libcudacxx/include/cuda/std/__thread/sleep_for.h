// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___THREAD_SLEEP_FOR_H
#define _CUDA_STD___THREAD_SLEEP_FOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__limits/numeric_limits.h>

#include <nv/target>

#if _CCCL_HOSTED()
#  if _CCCL_HAS_THREAD_API(PTHREAD)
#    include <cuda/std/__utility/cmp.h>
#    include <cuda/std/ctime>

#    include <errno.h> // IWYU pragma: keep
#  endif // _CCCL_HAS_THREAD_API(PTHREAD)
#  if _CCCL_HAS_THREAD_API(WIN32)
#    include <windows.h>
#  endif // _CCCL_HAS_THREAD_API(WIN32)
#endif // _CCCL_HOSTED()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_THREAD_API(PTHREAD)
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
#endif // _CCCL_HAS_THREAD_API(PTHREAD)

#if _CCCL_HOST_COMPILATION()
_CCCL_HOST_DEVICE_API inline void __cccl_thread_sleep_for_host(::cuda::std::chrono::nanoseconds __ns)
{
#  if _CCCL_HAS_THREAD_API(PTHREAD)
  auto __ts = ::cuda::std::__cccl_to_timespec(__ns);
  while (::nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
  {
  }
#  elif _CCCL_HAS_THREAD_API(WIN32)
  // round-up to the nearest millisecond
  chrono::milliseconds __ms = chrono::duration_cast<chrono::milliseconds>(__ns + chrono::nanoseconds(999999));
  ::Sleep(static_cast<DWORD>(__ms.count()));
#  endif // Unknown Thread API
}
#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_DEVICE_COMPILATION()
_CCCL_DEVICE_API inline void __cccl_thread_sleep_for_device(::cuda::std::chrono::nanoseconds __ns)
{
#  if _CCCL_HAS_THREAD_API(CUDA)
  NV_IF_TARGET(NV_PROVIDES_SM_70, ({
                 auto const __step = __ns.count();
                 _CCCL_ASSERT(__step < numeric_limits<unsigned>::max(), "invalid nanoseconds count");
                 ::__nanosleep((unsigned) __step);
               }))
#  elif _CCCL_CUDA_COMPILER(NVHPC)
  ::cuda::std::__cccl_thread_sleep_for_host(__ns);
#  endif // _CCCL_HAS_THREAD_API(CUDA)
}
#endif // _CCCL_DEVICE_COMPILATION()

_CCCL_HOST_DEVICE_API inline void __cccl_thread_sleep_for(::cuda::std::chrono::nanoseconds __ns)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (::cuda::std::__cccl_thread_sleep_for_host(__ns);),
                    (::cuda::std::__cccl_thread_sleep_for_device(__ns);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___THREAD_SLEEP_FOR_H
