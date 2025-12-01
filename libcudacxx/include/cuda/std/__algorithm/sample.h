//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_SAMPLE_H
#define _CUDA_STD___ALGORITHM_SAMPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__random/uniform_int_distribution.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_CCCL_API _SampleIterator __sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator& __g,
  input_iterator_tag)
{
  _Distance __k = 0;
  for (; __first != __last && __k < __n; ++__first, (void) ++__k)
  {
    __output_iter[__k] = *__first;
  }
  _Distance __sz = __k;
  for (; __first != __last; ++__first, (void) ++__k)
  {
    _Distance __r = ::cuda::std::uniform_int_distribution<_Distance>(0, __k)(__g);
    if (__r < __sz)
    {
      __output_iter[__r] = *__first;
    }
  }
  return __output_iter + ::cuda::std::min(__n, __k);
}

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_CCCL_API _SampleIterator __sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator& __g,
  forward_iterator_tag)
{
  _Distance __unsampled_sz = ::cuda::std::distance(__first, __last);
  for (__n = ::cuda::std::min(__n, __unsampled_sz); __n != 0; ++__first)
  {
    _Distance __r = ::cuda::std::uniform_int_distribution<_Distance>(0, --__unsampled_sz)(__g);
    if (__r < __n)
    {
      *__output_iter++ = *__first;
      --__n;
    }
  }
  return __output_iter;
}

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_CCCL_API _SampleIterator __sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator& __g)
{
  using _PopCategory = typename iterator_traits<_PopulationIterator>::iterator_category;
  using _Difference  = typename iterator_traits<_PopulationIterator>::difference_type;
  static_assert(__has_forward_traversal<_PopulationIterator> || __has_random_access_traversal<_SampleIterator>,
                "SampleIterator must meet the requirements of RandomAccessIterator");
  using _CommonType = typename common_type<_Distance, _Difference>::type;
  _CCCL_ASSERT(!is_signed_v<_Distance> || __n >= 0, "N must be a positive number.");
  return ::cuda::std::__sample(__first, __last, __output_iter, _CommonType(__n), __g, _PopCategory());
}

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_CCCL_API _SampleIterator sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator&& __g)
{
  return ::cuda::std::__sample(__first, __last, __output_iter, __n, __g);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_SAMPLE_H
