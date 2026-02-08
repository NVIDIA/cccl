//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_NTH_ELEMENT_H
#define _CUDA_STD___ALGORITHM_NTH_ELEMENT_H

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
#include <cuda/std/__algorithm/sort.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Compare, class _RandomAccessIterator>
_CCCL_API constexpr bool __nth_element_find_guard(
  _RandomAccessIterator& __i, _RandomAccessIterator& __j, _RandomAccessIterator __m, _Compare __comp)
{
  // manually guard downward moving __j against __i
  while (true)
  {
    if (__i == --__j)
    {
      return false;
    }
    if (__comp(*__j, *__m))
    {
      return true; // found guard for downward moving __j, now use unguarded partition
    }
  }
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_CCCL_API constexpr void
__nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp)
{
  using _Ops = _IterOps<_AlgPolicy>;

  // _Compare is known to be a reference type
  using difference_type         = typename iterator_traits<_RandomAccessIterator>::difference_type;
  const difference_type __limit = 7;
  while (true)
  {
    if (__nth == __last)
    {
      return;
    }
    difference_type __len = __last - __first;
    switch (__len)
    {
      case 0:
      case 1:
        return;
      case 2:
        if (__comp(*--__last, *__first))
        {
          _Ops::iter_swap(__first, __last);
        }
        return;
      case 3: {
        _RandomAccessIterator __m = __first;
        ::cuda::std::__sort3<_AlgPolicy, _Compare>(__first, ++__m, --__last, __comp);
        return;
      }
    }
    if (__len <= __limit)
    {
      ::cuda::std::__selection_sort<_AlgPolicy, _Compare>(__first, __last, __comp);
      return;
    }
    // __len > __limit >= 3
    _RandomAccessIterator __m   = __first + __len / 2;
    _RandomAccessIterator __lm1 = __last;
    unsigned __n_swaps          = ::cuda::std::__sort3<_AlgPolicy, _Compare>(__first, __m, --__lm1, __comp);
    // *__m is median
    // partition [__first, __m) < *__m and *__m <= [__m, __last)
    // (this inhibits tossing elements equivalent to __m around unnecessarily)
    _RandomAccessIterator __i = __first;
    _RandomAccessIterator __j = __lm1;
    // j points beyond range to be tested, *__lm1 is known to be <= *__m
    // The search going up is known to be guarded but the search coming down isn't.
    // Prime the downward search with a guard.
    if (!__comp(*__i, *__m)) // if *__first == *__m
    {
      // *__first == *__m, *__first doesn't go in first part
      if (::cuda::std::__nth_element_find_guard<_Compare>(__i, __j, __m, __comp))
      {
        _Ops::iter_swap(__i, __j);
        ++__n_swaps;
      }
      else
      {
        // *__first == *__m, *__m <= all other elements
        // Partition instead into [__first, __i) == *__first and *__first < [__i, __last)
        ++__i; // __first + 1
        __j = __last;
        if (!__comp(*__first, *--__j))
        { // we need a guard if *__first == *(__last-1)
          while (true)
          {
            if (__i == __j)
            {
              return; // [__first, __last) all equivalent elements
            }
            else if (__comp(*__first, *__i))
            {
              _Ops::iter_swap(__i, __j);
              ++__n_swaps;
              ++__i;
              break;
            }
            ++__i;
          }
        }
        // [__first, __i) == *__first and *__first < [__j, __last) and __j == __last - 1
        if (__i == __j)
        {
          return;
        }
        while (true)
        {
          while (!__comp(*__first, *__i))
          {
            ++__i;
            _CCCL_ASSERT(__i != __last,
                         "Would read out of bounds, does your comparator satisfy the strict-weak ordering "
                         "requirement?");
          }
          do
          {
            _CCCL_ASSERT(__j != __first,
                         "Would read out of bounds, does your comparator satisfy the strict-weak ordering "
                         "requirement?");
            --__j;
          } while (__comp(*__first, *__j));
          if (__i >= __j)
          {
            break;
          }
          _Ops::iter_swap(__i, __j);
          ++__n_swaps;
          ++__i;
        }
        // [__first, __i) == *__first and *__first < [__i, __last)
        // The first part is sorted,
        if (__nth < __i)
        {
          return;
        }
        // __nth_element the second part
        // ::cuda::std::__nth_element<_Compare>(__i, __nth, __last, __comp);
        __first = __i;
        continue;
      }
    }
    ++__i;
    // j points beyond range to be tested, *__lm1 is known to be <= *__m
    // if not yet partitioned...
    if (__i < __j)
    {
      // known that *(__i - 1) < *__m
      while (true)
      {
        // __m still guards upward moving __i
        while (__comp(*__i, *__m))
        {
          ++__i;
          _CCCL_ASSERT(__i != __last,
                       "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
        }
        // It is now known that a guard exists for downward moving __j
        do
        {
          _CCCL_ASSERT(__j != __first,
                       "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
          --__j;
        } while (!__comp(*__j, *__m));
        if (__i >= __j)
        {
          break;
        }
        _Ops::iter_swap(__i, __j);
        ++__n_swaps;
        // It is known that __m != __j
        // If __m just moved, follow it
        if (__m == __i)
        {
          __m = __j;
        }
        ++__i;
      }
    }
    // [__first, __i) < *__m and *__m <= [__i, __last)
    if (__i != __m && __comp(*__m, *__i))
    {
      _Ops::iter_swap(__i, __m);
      ++__n_swaps;
    }
    // [__first, __i) < *__i and *__i <= [__i+1, __last)
    if (__nth == __i)
    {
      return;
    }
    if (__n_swaps == 0)
    {
      // We were given a perfectly partitioned sequence.  Coincidence?
      if (__nth < __i)
      {
        // Check for [__first, __i) already sorted
        __j = __m = __first;
        while (true)
        {
          if (++__j == __i)
          {
            // [__first, __i) sorted
            return;
          }
          if (__comp(*__j, *__m))
          {
            // not yet sorted, so sort
            break;
          }
          __m = __j;
        }
      }
      else
      {
        // Check for [__i, __last) already sorted
        __j = __m = __i;
        while (true)
        {
          if (++__j == __last)
          {
            // [__i, __last) sorted
            return;
          }
          if (__comp(*__j, *__m))
          {
            // not yet sorted, so sort
            break;
          }
          __m = __j;
        }
      }
    }
    // __nth_element on range containing __nth
    if (__nth < __i)
    {
      // ::cuda::std::__nth_element<_Compare>(__first, __nth, __i, __comp);
      __last = __i;
    }
    else
    {
      // ::cuda::std::__nth_element<_Compare>(__i+1, __nth, __last, __comp);
      __first = ++__i;
    }
  }
}

template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr void __nth_element_impl(
  _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare& __comp)
{
  if (__nth == __last)
  {
    return;
  }
  ::cuda::std::__nth_element<_AlgPolicy, __comp_ref_type<_Compare>>(__first, __nth, __last, __comp);
}

template <class _RandomAccessIterator, class _Compare>
_CCCL_API constexpr void
nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp)
{
  ::cuda::std::__nth_element_impl<_ClassicAlgPolicy>(
    ::cuda::std::move(__first), ::cuda::std::move(__nth), ::cuda::std::move(__last), __comp);
}

template <class _RandomAccessIterator>
_CCCL_API constexpr void
nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last)
{
  ::cuda::std::nth_element(::cuda::std::move(__first), ::cuda::std::move(__nth), ::cuda::std::move(__last), __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_NTH_ELEMENT_H
