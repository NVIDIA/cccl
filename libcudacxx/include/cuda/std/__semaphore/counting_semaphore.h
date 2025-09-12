//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SEMAPHORE_COUNTING_SEMAPHORE_H
#define _CUDA_STD___SEMAPHORE_COUNTING_SEMAPHORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__semaphore/atomic_semaphore.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <ptrdiff_t __least_max_value = INT_MAX>
class counting_semaphore : public __atomic_semaphore<thread_scope_system, __least_max_value>
{
  static_assert(__least_max_value <= __atomic_semaphore<thread_scope_system, __least_max_value>::max(), "");

public:
  _CCCL_API constexpr counting_semaphore(ptrdiff_t __count = 0)
      : __atomic_semaphore<thread_scope_system, __least_max_value>(__count)
  {}
  _CCCL_HIDE_FROM_ABI ~counting_semaphore() = default;

  counting_semaphore(const counting_semaphore&)            = delete;
  counting_semaphore& operator=(const counting_semaphore&) = delete;
};

using binary_semaphore = counting_semaphore<1>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SEMAPHORE_COUNTING_SEMAPHORE_H
