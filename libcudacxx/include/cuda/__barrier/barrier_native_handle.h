//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_BARRIER_NATIVE_HANDLE_H
#define _CUDA___BARRIER_BARRIER_NATIVE_HANDLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_block_scope.h>
#include <cuda/__fwd/barrier.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

_CCCL_DEVICE inline _CUDA_VSTD::uint64_t* barrier_native_handle(barrier<thread_scope_block>& __b)
{
  return reinterpret_cast<_CUDA_VSTD::uint64_t*>(&__b.__barrier);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CUDA___BARRIER_BARRIER_NATIVE_HANDLE_H
