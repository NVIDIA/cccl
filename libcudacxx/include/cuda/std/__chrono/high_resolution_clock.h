// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_HIGH_RESOLUTION_CLOCK_H
#define _CUDA_STD___CHRONO_HIGH_RESOLUTION_CLOCK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/steady_clock.h>
#include <cuda/std/__chrono/system_clock.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
#if _LIBCUDACXX_HAS_MONOTONIC_CLOCK()
using high_resolution_clock = steady_clock;
#else // ^^^ _LIBCUDACXX_HAS_MONOTONIC_CLOCK() ^^^ / vvv !_LIBCUDACXX_HAS_MONOTONIC_CLOCK() vvv
using high_resolution_clock = system_clock;
#endif // !_LIBCUDACXX_HAS_MONOTONIC_CLOCK()
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_HIGH_RESOLUTION_CLOCK_H
