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

using _CUDA_VRANGES::iota_iterator;

//! @brief make_transform_iterator creates a \p transform_iterator from an \c Iterator and \c _Fn.
//!
//! @param __iter The \c Iterator pointing to the input range of the newly created \p transform_iterator.
//! @param __fun The \c _Fn used to transform the range pointed to by @param __iter in the newly created
//! \p transform_iterator.
//! @return A new \p transform_iterator which transforms the range at @param __iter by @param __fun.
template <class _Start>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto make_iota_iterator(_Start __start)
{
  return iota_iterator{__start};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_IOTA_ITERATOR_H
