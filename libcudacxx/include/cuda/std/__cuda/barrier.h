// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_BARRIER_H
#define _LIBCUDACXX___CUDA_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 700
#  error "CUDA synchronization primitives are only supported for sm_70 and up."
#endif

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/aligned_size.h>
#include <cuda/__barrier/barrier.h>
#include <cuda/__barrier/barrier_arrive_tx.h>
#include <cuda/__barrier/barrier_expect_tx.h>
#include <cuda/__barrier/barrier_thread_scope_block.h>
#include <cuda/__barrier/barrier_thread_scope_thread.h>
#include <cuda/__fwd/pipeline.h>
#include <cuda/__memcpy_async/memcpy_async.h>
#include <cuda/__memcpy_async/memcpy_async_tx.h>

#if defined(_CCCL_CUDA_COMPILER)
#  include <cuda/ptx> // cuda::ptx::*
#endif // _CCCL_CUDA_COMPILER

#if defined(_CCCL_CUDA_COMPILER)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_CUDA_COMPILER

#endif // _LIBCUDACXX___CUDA_BARRIER_H
