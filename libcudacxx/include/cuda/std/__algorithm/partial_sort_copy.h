//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_PARTIAL_SORT_COPY_H
#define _CUDA_STD___ALGORITHM_PARTIAL_SORT_COPY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/comp_ref_type.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/make_heap.h>
#include <cuda/std/__algorithm/make_projected.h>
#include <cuda/std/__algorithm/sift_down.h>
#include <cuda/std/__algorithm/sort_heap.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy,
          class _Compare,
          class _InputIterator,
          class _Sentinel1,
          class _RandomAccessIterator,
          class _Sentinel2,
          class _Proj1,
          class _Proj2>
_CCCL_API constexpr pair<_InputIterator, _RandomAccessIterator> __partial_sort_copy(
  _InputIterator __first,
  _Sentinel1 __last,
  _RandomAccessIterator __result_first,
  _Sentinel2 __result_last,
  _Compare&& __comp,
  _Proj1&& __proj1,
  _Proj2&& __proj2)
{
  _RandomAccessIterator __r = __result_first;
  auto&& __projected_comp   = ::cuda::std::__make_projected(__comp, __proj2);

  if (__r != __result_last)
  {
    for (; __first != __last && __r != __result_last; ++__first, (void) ++__r)
    {
      *__r = *__first;
    }
    ::cuda::std::__make_heap<_AlgPolicy>(__result_first, __r, __projected_comp);
    typename iterator_traits<_RandomAccessIterator>::difference_type __len = __r - __result_first;
    for (; __first != __last; ++__first)
    {
      if (::cuda::std::invoke(
            __comp, ::cuda::std::invoke(__proj1, *__first), ::cuda::std::invoke(__proj2, *__result_first)))
      {
        *__result_first = *__first;
        ::cuda::std::__sift_down<_AlgPolicy>(__result_first, __projected_comp, __len, __result_first);
      }
    }
    ::cuda::std::__sort_heap<_AlgPolicy>(__result_first, __r, __projected_comp);
  }

  return pair<_InputIterator, _RandomAccessIterator>(
    _IterOps<_AlgPolicy>::next(::cuda::std::move(__first), ::cuda::std::move(__last)), ::cuda::std::move(__r));
}

template <class _InputIterator, class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr _RandomAccessIterator partial_sort_copy(
  _InputIterator __first,
  _InputIterator __last,
  _RandomAccessIterator __result_first,
  _RandomAccessIterator __result_last,
  _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first), decltype(*__result_first)>::value,
                "Comparator has to be callable");

  auto __result = ::cuda::std::__partial_sort_copy<_ClassicAlgPolicy>(
    __first,
    __last,
    __result_first,
    __result_last,
    static_cast<__comp_ref_type<_Compare>>(__comp),
    identity(),
    identity());
  return __result.second;
}

template <class _InputIterator, class _RandomAccessIterator>
_CCCL_API constexpr _RandomAccessIterator partial_sort_copy(
  _InputIterator __first,
  _InputIterator __last,
  _RandomAccessIterator __result_first,
  _RandomAccessIterator __result_last)
{
  return ::cuda::std::partial_sort_copy(__first, __last, __result_first, __result_last, __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_PARTIAL_SORT_COPY_H
