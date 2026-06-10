// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_CALENDAR_H
#define _CUDA_STD___CHRONO_CALENDAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/time_point.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
struct local_t
{};

template <class Duration>
using local_time    = time_point<local_t, Duration>;
using local_seconds = local_time<seconds>;
using local_days    = local_time<days>;

struct _CCCL_TYPE_VISIBILITY_DEFAULT last_spec
{
  _CCCL_HIDE_FROM_ABI explicit last_spec() = default;
};
inline constexpr last_spec last{};
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_CALENDAR_H
