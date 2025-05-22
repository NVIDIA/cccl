// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_IOTA_ITERATOR_H
#define _CUDA___ITERATOR_IOTA_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <cuda/std/__ranges/iota_iterator.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Start>
using counting_iterator = _CUDA_VRANGES::__iota_iterator<_Start>;

//! @brief make_counting_iterator creates a \p counting_iterator from an __integer-like__ \c _Start
//! @param __start The __integer-like__ \c _Start representing the initial count
template <class _Start>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto make_counting_iterator(_Start __start)
{
  return counting_iterator<_Start>{__start};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_IOTA_ITERATOR_H
