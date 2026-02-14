//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___FORWARD_ZIP_ITERATOR_H
#define _CUDA___FORWARD_ZIP_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class... _Iterators>
class zip_iterator;

template <class>
inline constexpr bool __is_zip_iterator = false;

template <class... _Iterators>
inline constexpr bool __is_zip_iterator<zip_iterator<_Iterators...>> = true;

template <class _Fn>
class zip_function;

template <class>
inline constexpr bool __is_zip_function = false;

template <class _Fn>
inline constexpr bool __is_zip_function<zip_function<_Fn>> = true;

template <class _Fn, class... _Iterators>
class zip_transform_iterator;

template <class>
inline constexpr bool __is_zip_transform_iterator = false;

template <class _Fn, class... _Iterators>
inline constexpr bool __is_zip_transform_iterator<zip_transform_iterator<_Fn, _Iterators...>> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FORWARD_ZIP_ITERATOR_H
