// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_STEADY_CLOCK_H
#define _CUDA_STD___CHRONO_STEADY_CLOCK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_MONOTONIC_CLOCK()

#  include <cuda/std/__chrono/duration.h>
#  include <cuda/std/__chrono/time_point.h>

#  if !_CCCL_COMPILER(NVRTC)
#    include <chrono>
#  endif // !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
class _CCCL_TYPE_VISIBILITY_DEFAULT steady_clock
{
public:
  using duration                  = nanoseconds;
  using rep                       = duration::rep;
  using period                    = duration::period;
  using time_point                = ::cuda::std::chrono::time_point<steady_clock, duration>;
  static constexpr bool is_steady = true;

  [[nodiscard]] _CCCL_API static time_point now() noexcept;
};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_HAS_MONOTONIC_CLOCK()

#endif // _CUDA_STD___CHRONO_STEADY_CLOCK_H
