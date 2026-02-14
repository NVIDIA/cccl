//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_PARTIAL_SORT_H
#define _CUDA_STD___ALGORITHM_PARTIAL_SORT_H

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
#include <cuda/std/__algorithm/sift_down.h>
#include <cuda/std/__algorithm/sort_heap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Sentinel>
_CCCL_API constexpr _RandomAccessIterator
__partial_sort_impl(_RandomAccessIterator __first, _RandomAccessIterator __middle, _Sentinel __last, _Compare&& __comp)
{
  if (__first == __middle)
  {
    return _IterOps<_AlgPolicy>::next(__middle, __last);
  }

  ::cuda::std::__make_heap<_AlgPolicy>(__first, __middle, __comp);

  typename iterator_traits<_RandomAccessIterator>::difference_type __len = __middle - __first;
  _RandomAccessIterator __i                                              = __middle;
  for (; __i != __last; ++__i)
  {
    if (__comp(*__i, *__first))
    {
      _IterOps<_AlgPolicy>::iter_swap(__i, __first);
      ::cuda::std::__sift_down<_AlgPolicy>(__first, __comp, __len, __first);
    }
  }
  ::cuda::std::__sort_heap<_AlgPolicy>(::cuda::std::move(__first), ::cuda::std::move(__middle), __comp);

  return __i;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Sentinel>
_CCCL_API constexpr _RandomAccessIterator
__partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _Sentinel __last, _Compare& __comp)
{
  if (__first == __middle)
  {
    return _IterOps<_AlgPolicy>::next(__middle, __last);
  }

  return ::cuda::std::__partial_sort_impl<_AlgPolicy>(
    __first, __middle, __last, static_cast<__comp_ref_type<_Compare>>(__comp));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr void partial_sort(
  _RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp)
{
  static_assert(is_copy_constructible_v<_RandomAccessIterator>, "Iterators must be copy constructible.");
  static_assert(is_copy_assignable_v<_RandomAccessIterator>, "Iterators must be copy assignable.");

  (void) ::cuda::std::__partial_sort<_ClassicAlgPolicy>(
    ::cuda::std::move(__first), ::cuda::std::move(__middle), ::cuda::std::move(__last), __comp);
}

template <class _RandomAccessIterator>
_CCCL_API constexpr void
partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last)
{
  ::cuda::std::partial_sort(__first, __middle, __last, __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_PARTIAL_SORT_H
