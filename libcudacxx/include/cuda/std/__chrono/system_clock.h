// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_SYSTEM_CLOCK_H
#define _CUDA_STD___CHRONO_SYSTEM_CLOCK_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/ctime>

#include <nv/target>

// Android before API level 29 has no `timespec_get` (see <cuda/std/ctime>). Only there the host time comes from
// <chrono>.
#if _CCCL_HOSTED() && defined(__ANDROID_API__) && __ANDROID_API__ < 29
#  include <chrono> // IWYU pragma: keep
#endif // _CCCL_HOSTED() && defined(__ANDROID_API__) && __ANDROID_API__ < 29

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class _CCCL_TYPE_VISIBILITY_DEFAULT system_clock
{
public:
  using duration                  = ::cuda::std::chrono::nanoseconds;
  using rep                       = duration::rep;
  using period                    = duration::period;
  using time_point                = ::cuda::std::chrono::time_point<system_clock>;
  static constexpr bool is_steady = false;

  [[nodiscard]] _CCCL_HOST_DEVICE_API static time_point now() noexcept
  {
#if !defined(__ANDROID_API__) || __ANDROID_API__ >= 29
    // The host time comes from `timespec_get` and not from `::std::chrono::system_clock`. The host's <chrono> is
    // expensive to compile (in C++20 it includes <format>), and <cuda/std/atomic> reaches this header.
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (::cuda::std::timespec __ts{}; (void) ::cuda::std::timespec_get(&__ts, TIME_UTC);
                       return time_point(duration_cast<duration>(seconds(__ts.tv_sec) + nanoseconds(__ts.tv_nsec)));),
                      (return time_point(duration_cast<duration>(nanoseconds(::cuda::ptx::get_sreg_globaltimer())));))
#else // ^^^ has timespec_get ^^^ / vvv no timespec_get vvv
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (return time_point(duration_cast<duration>(nanoseconds(
        ::std::chrono::duration_cast<::std::chrono::nanoseconds>(::std::chrono::system_clock::now().time_since_epoch())
          .count())));),
      (return time_point(duration_cast<duration>(nanoseconds(::cuda::ptx::get_sreg_globaltimer())));))
#endif // no timespec_get
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static time_t to_time_t(const time_point& __t) noexcept
  {
    return time_t(::cuda::std::chrono::duration_cast<seconds>(__t.time_since_epoch()).count());
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static time_point from_time_t(time_t __t) noexcept
  {
    return time_point(::cuda::std::chrono::seconds(__t));
  }
};

template <class _Duration>
using sys_time    = time_point<system_clock, _Duration>;
using sys_seconds = sys_time<seconds>;
using sys_days    = sys_time<days>;
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_SYSTEM_CLOCK_H
