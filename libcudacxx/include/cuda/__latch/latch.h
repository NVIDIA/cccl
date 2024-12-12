// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__LATCH_LATCH_H
#define __CUDA__LATCH_LATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__latch/latch.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <thread_scope _Sco>
class latch : public _CUDA_VSTD::__latch_base<_Sco>
{
public:
  _LIBCUDACXX_HIDE_FROM_ABI constexpr latch(_CUDA_VSTD::ptrdiff_t __count)
      : _CUDA_VSTD::__latch_base<_Sco>(__count)
  {}
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // __CUDA__LATCH_LATCH_H
