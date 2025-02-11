//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_SAMPLE_H
#define _LIBCUDACXX___ALGORITHM_SAMPLE_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_LIBCUDACXX_HIDE_FROM_ABI _SampleIterator __sample(
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
    _Distance __r = _CUDA_VSTD::uniform_int_distribution<_Distance>(0, __k)(__g);
    if (__r < __sz)
    {
      __output_iter[__r] = *__first;
    }
  }
  return __output_iter + _CUDA_VSTD::min(__n, __k);
}

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_LIBCUDACXX_HIDE_FROM_ABI _SampleIterator __sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator& __g,
  forward_iterator_tag)
{
  _Distance __unsampled_sz = _CUDA_VSTD::distance(__first, __last);
  for (__n = _CUDA_VSTD::min(__n, __unsampled_sz); __n != 0; ++__first)
  {
    _Distance __r = _CUDA_VSTD::uniform_int_distribution<_Distance>(0, --__unsampled_sz)(__g);
    if (__r < __n)
    {
      *__output_iter++ = *__first;
      --__n;
    }
  }
  return __output_iter;
}

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_LIBCUDACXX_HIDE_FROM_ABI _SampleIterator __sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator& __g)
{
  typedef typename iterator_traits<_PopulationIterator>::iterator_category _PopCategory;
  typedef typename iterator_traits<_PopulationIterator>::difference_type _Difference;
  static_assert(__is_cpp17_forward_iterator<_PopulationIterator>::value
                  || __is_cpp17_random_access_iterator<_SampleIterator>::value,
                "SampleIterator must meet the requirements of RandomAccessIterator");
  typedef typename common_type<_Distance, _Difference>::type _CommonType;
  _CCCL_ASSERT(!_CCCL_TRAIT(is_signed, _Distance) || __n >= 0, "N must be a positive number.");
  return _CUDA_VSTD::__sample(__first, __last, __output_iter, _CommonType(__n), __g, _PopCategory());
}

template <class _PopulationIterator, class _SampleIterator, class _Distance, class _UniformRandomNumberGenerator>
_LIBCUDACXX_HIDE_FROM_ABI _SampleIterator sample(
  _PopulationIterator __first,
  _PopulationIterator __last,
  _SampleIterator __output_iter,
  _Distance __n,
  _UniformRandomNumberGenerator&& __g)
{
  return _CUDA_VSTD::__sample(__first, __last, __output_iter, __n, __g);
}

template <class _RandomAccessIterator, class _UniformRandomNumberGenerator>
_LIBCUDACXX_HIDE_FROM_ABI void
shuffle(_RandomAccessIterator __first, _RandomAccessIterator __last, _UniformRandomNumberGenerator&& __g)
{
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  typedef uniform_int_distribution<ptrdiff_t> _Dp;
  typedef typename _Dp::param_type _Pp;
  difference_type __d = __last - __first;
  if (__d > 1)
  {
    _Dp __uid;
    for (--__last, (void) --__d; __first < __last; ++__first, (void) --__d)
    {
      difference_type __i = __uid(__g, _Pp(0, __d));
      if (__i != difference_type(0))
      {
        swap(*__first, *(__first + __i));
      }
    }
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_SAMPLE_H
