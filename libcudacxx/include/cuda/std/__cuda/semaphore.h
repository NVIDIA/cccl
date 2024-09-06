// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_SEMAPHORE_H
#define _LIBCUDACXX___CUDA_SEMAPHORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <thread_scope _Sco, ptrdiff_t __least_max_value = INT_MAX>
class counting_semaphore : public _CUDA_VSTD::__semaphore_base<__least_max_value, _Sco>
{
  static_assert(__least_max_value <= _CUDA_VSTD::__semaphore_base<__least_max_value, _Sco>::max(), "");

public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr counting_semaphore(ptrdiff_t __count = 0)
      : _CUDA_VSTD::__semaphore_base<__least_max_value, _Sco>(__count)
  {}
  _CCCL_HIDE_FROM_ABI ~counting_semaphore() = default;

  counting_semaphore(const counting_semaphore&)            = delete;
  counting_semaphore& operator=(const counting_semaphore&) = delete;
};

template <thread_scope _Sco>
using binary_semaphore = counting_semaphore<_Sco, 1>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_SEMAPHORE_H
