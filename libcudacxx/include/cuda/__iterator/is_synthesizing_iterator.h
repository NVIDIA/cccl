//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_IS_SYNTHESIZING_ITERATOR_H
#define _CUDA___ITERATOR_IS_SYNTHESIZING_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/iterator.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// True when dereferencing `_Iter` produces a value without loading memory outside the iterator.
// `constant_iterator`, `counting_iterator`, and `shuffle_iterator` store everything required to produce their value.
// Adaptor iterators are synthesizing when every iterator they read is synthesizing. Thrust equivalents specialize this
// variable after their class definitions. cv-qualifiers and references are ignored.
template <class _Iter>
inline constexpr bool __is_synthesizing_iterator_v = false;

template <class _Iter>
inline constexpr bool __is_synthesizing_iterator_v<const _Iter> = __is_synthesizing_iterator_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_synthesizing_iterator_v<volatile _Iter> = __is_synthesizing_iterator_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_synthesizing_iterator_v<const volatile _Iter> = __is_synthesizing_iterator_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_synthesizing_iterator_v<_Iter&> = __is_synthesizing_iterator_v<_Iter>;

template <class _Iter>
inline constexpr bool __is_synthesizing_iterator_v<_Iter&&> = __is_synthesizing_iterator_v<_Iter>;

template <class _Tp, class _Index>
inline constexpr bool __is_synthesizing_iterator_v<constant_iterator<_Tp, _Index>> = true;

template <class _Start, class _DiffT>
inline constexpr bool __is_synthesizing_iterator_v<counting_iterator<_Start, _DiffT>> = true;

template <class _IndexType, class _Bijection>
inline constexpr bool __is_synthesizing_iterator_v<shuffle_iterator<_IndexType, _Bijection>> = true;

template <class _Iter, class _Stride>
inline constexpr bool __is_synthesizing_iterator_v<strided_iterator<_Iter, _Stride>> =
  __is_synthesizing_iterator_v<_Iter>;

template <class _Fn, class _Iter>
inline constexpr bool __is_synthesizing_iterator_v<transform_iterator<_Fn, _Iter>> =
  __is_synthesizing_iterator_v<_Iter>;

template <class... _Iterators>
inline constexpr bool __is_synthesizing_iterator_v<zip_iterator<_Iterators...>> =
  (__is_synthesizing_iterator_v<_Iterators> && ...);

template <class _Fn, class... _Iterators>
inline constexpr bool __is_synthesizing_iterator_v<zip_transform_iterator<_Fn, _Iterators...>> =
  (__is_synthesizing_iterator_v<_Iterators> && ...);

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_IS_SYNTHESIZING_ITERATOR_H
