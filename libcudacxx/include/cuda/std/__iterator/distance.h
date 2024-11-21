// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_DISTANCE_H
#define _LIBCUDACXX___ITERATOR_DISTANCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/remove_cvref.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 typename iterator_traits<_InputIter>::difference_type
__distance(_InputIter __first, _InputIter __last, input_iterator_tag)
{
  typename iterator_traits<_InputIter>::difference_type __r(0);
  for (; __first != __last; ++__first)
  {
    ++__r;
  }
  return __r;
}

template <class _RandIter>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 typename iterator_traits<_RandIter>::difference_type
__distance(_RandIter __first, _RandIter __last, random_access_iterator_tag)
{
  return __last - __first;
}

template <class _InputIter>
_LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 typename iterator_traits<_InputIter>::difference_type
distance(_InputIter __first, _InputIter __last)
{
  return _CUDA_VSTD::__distance(__first, __last, typename iterator_traits<_InputIter>::iterator_category());
}

_LIBCUDACXX_END_NAMESPACE_STD
#if _CCCL_STD_VER > 2014 && !_CCCL_COMPILER(MSVC2017)

// [range.iter.op.distance]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__distance)
struct __fn
{
  _CCCL_TEMPLATE(class _Ip, class _Sp)
  _CCCL_REQUIRES((sentinel_for<_Sp, _Ip> && !sized_sentinel_for<_Sp, _Ip>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr iter_difference_t<_Ip> operator()(_Ip __first, _Sp __last) const
  {
    iter_difference_t<_Ip> __n = 0;
    while (__first != __last)
    {
      ++__first;
      ++__n;
    }
    return __n;
  }

  _CCCL_TEMPLATE(class _Ip, class _Sp)
  _CCCL_REQUIRES((sized_sentinel_for<_Sp, decay_t<_Ip>>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr iter_difference_t<_Ip> operator()(_Ip&& __first, _Sp __last) const
  {
    if constexpr (sized_sentinel_for<_Sp, remove_cvref_t<_Ip>>)
    {
      return __last - __first;
    }
    else
    {
      return __last - decay_t<_Ip>(__first);
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES((range<_Rp>) )
  _LIBCUDACXX_HIDE_FROM_ABI constexpr range_difference_t<_Rp> operator()(_Rp&& __r) const
  {
    if constexpr (sized_range<_Rp>)
    {
      return static_cast<range_difference_t<_Rp>>(_CUDA_VRANGES::size(__r));
    }
    else
    {
      return operator()(_CUDA_VRANGES::begin(__r), _CUDA_VRANGES::end(__r));
    }
    _CCCL_UNREACHABLE();
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto distance = __distance::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER > 2014  && !_CCCL_COMPILER(MSVC2017)

#endif // _LIBCUDACXX___ITERATOR_DISTANCE_H
