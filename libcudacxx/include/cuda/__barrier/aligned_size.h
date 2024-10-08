//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_ALIGNED_SIZE_H
#define _CUDA___BARRIER_ALIGNED_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <_CUDA_VSTD::size_t _Alignment>
struct aligned_size_t
{
  static constexpr _CUDA_VSTD::size_t align = _Alignment;
  _CUDA_VSTD::size_t value;

  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr aligned_size_t(size_t __s)
      : value(__s)
  {}
  _LIBCUDACXX_HIDE_FROM_ABI constexpr operator size_t() const
  {
    return value;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_ALIGNED_SIZE_H
