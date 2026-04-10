//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FWD_ITERATOR_H
#define _CUDA___FWD_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/random.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, class _Index = ::cuda::std::ptrdiff_t>
class constant_iterator;

#if _CCCL_HAS_CONCEPTS()
template <::cuda::std::weakly_incrementable _Start>
  requires ::cuda::std::copyable<_Start>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Start,
          ::cuda::std::enable_if_t<::cuda::std::weakly_incrementable<_Start>, int> = 0,
          ::cuda::std::enable_if_t<::cuda::std::copyable<_Start>, int>             = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class counting_iterator;

class discard_iterator;

template <class _Iter, class _Index = _Iter>
class permutation_iterator;

template <class _IndexType = ::cuda::std::size_t, class _Bijection = random_bijection<_IndexType>>
class shuffle_iterator;

template <class _Iter, class _Stride = ::cuda::std::iter_difference_t<_Iter>>
class strided_iterator;

template <class _Fn, class _Index = ::cuda::std::ptrdiff_t>
class tabulate_output_iterator;

template <class _InputFn, class _OutputFn, class _Iter>
class transform_input_output_iterator;

template <class _Fn, class _Iter>
class transform_iterator;

template <class _Fn, class _Iter>
class transform_output_iterator;

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

#endif // _CUDA___FWD_ITERATOR_H
