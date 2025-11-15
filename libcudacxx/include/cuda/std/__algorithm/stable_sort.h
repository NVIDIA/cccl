//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_STABLE_SORT_H
#define _CUDA_STD___ALGORITHM_STABLE_SORT_H

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
#include <cuda/std/__algorithm/inplace_merge.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/sort.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__memory/destruct_n.h>
#include <cuda/std/__memory/temporary_buffer.h>
#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _AlgPolicy, class _Compare, class _BidirectionalIterator>
_CCCL_API void __insertion_sort_move(
  _BidirectionalIterator __first1,
  _BidirectionalIterator __last1,
  typename iterator_traits<_BidirectionalIterator>::value_type* __first2,
  _Compare __comp)
{
  using _Ops       = _IterOps<_AlgPolicy>;
  using value_type = typename iterator_traits<_BidirectionalIterator>::value_type;
  if (__first1 != __last1)
  {
    __destruct_n __d(0);
    unique_ptr<value_type, __destruct_n&> __h(__first2, __d);
    value_type* __last2 = __first2;
    ::cuda::std::__construct_at(__last2, _Ops::__iter_move(__first1));
    __d.template __incr<value_type>();
    for (++__last2; ++__first1 != __last1; ++__last2)
    {
      value_type* __j2 = __last2;
      value_type* __i2 = __j2;
      if (__comp(*__first1, *--__i2))
      {
        ::cuda::std::__construct_at(__j2, ::cuda::std::move(*__i2));
        __d.template __incr<value_type>();
        for (--__j2; __i2 != __first2 && __comp(*__first1, *--__i2); --__j2)
        {
          *__j2 = ::cuda::std::move(*__i2);
        }
        *__j2 = _Ops::__iter_move(__first1);
      }
      else
      {
        ::cuda::std::__construct_at(__j2, _Ops::__iter_move(__first1));
        __d.template __incr<value_type>();
      }
    }
    __h.release();
  }
}

template <class _AlgPolicy, class _Compare, class _InputIterator1, class _InputIterator2>
_CCCL_API void __merge_move_construct(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  typename iterator_traits<_InputIterator1>::value_type* __result,
  _Compare __comp)
{
  using _Ops       = _IterOps<_AlgPolicy>;
  using value_type = typename iterator_traits<_InputIterator1>::value_type;

  __destruct_n __d(0);
  unique_ptr<value_type, __destruct_n&> __h(__result, __d);
  for (; true; ++__result)
  {
    if (__first1 == __last1)
    {
      for (; __first2 != __last2; ++__first2, (void) ++__result, __d.template __incr<value_type>())
      {
        ::cuda::std::__construct_at(__result, _Ops::__iter_move(__first2));
      }
      __h.release();
      return;
    }
    if (__first2 == __last2)
    {
      for (; __first1 != __last1; ++__first1, (void) ++__result, __d.template __incr<value_type>())
      {
        ::cuda::std::__construct_at(__result, _Ops::__iter_move(__first1));
      }
      __h.release();
      return;
    }
    if (__comp(*__first2, *__first1))
    {
      ::cuda::std::__construct_at(__result, _Ops::__iter_move(__first2));
      __d.template __incr<value_type>();
      ++__first2;
    }
    else
    {
      ::cuda::std::__construct_at(__result, _Ops::__iter_move(__first1));
      __d.template __incr<value_type>();
      ++__first1;
    }
  }
}

template <class _AlgPolicy, class _Compare, class _InputIterator1, class _InputIterator2, class _OutputIterator>
_CCCL_API void __merge_move_assign(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  _OutputIterator __result,
  _Compare __comp)
{
  using _Ops = _IterOps<_AlgPolicy>;

  for (; __first1 != __last1; ++__result)
  {
    if (__first2 == __last2)
    {
      for (; __first1 != __last1; ++__first1, (void) ++__result)
      {
        *__result = _Ops::__iter_move(__first1);
      }
      return;
    }
    if (__comp(*__first2, *__first1))
    {
      *__result = _Ops::__iter_move(__first2);
      ++__first2;
    }
    else
    {
      *__result = _Ops::__iter_move(__first1);
      ++__first1;
    }
  }
  for (; __first2 != __last2; ++__first2, (void) ++__result)
  {
    *__result = _Ops::__iter_move(__first2);
  }
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_CCCL_API void __stable_sort(
  _RandomAccessIterator __first,
  _RandomAccessIterator __last,
  _Compare __comp,
  typename iterator_traits<_RandomAccessIterator>::difference_type __len,
  typename iterator_traits<_RandomAccessIterator>::value_type* __buff,
  ptrdiff_t __buff_size);

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_CCCL_API void __stable_sort_move(
  _RandomAccessIterator __first1,
  _RandomAccessIterator __last1,
  _Compare __comp,
  typename iterator_traits<_RandomAccessIterator>::difference_type __len,
  typename iterator_traits<_RandomAccessIterator>::value_type* __first2)
{
  using _Ops       = _IterOps<_AlgPolicy>;
  using value_type = typename iterator_traits<_RandomAccessIterator>::value_type;

  switch (__len)
  {
    case 0:
      return;
    case 1:
      ::cuda::std::__construct_at(__first2, _Ops::__iter_move(__first1));
      return;
    case 2:
      __destruct_n __d(0);
      unique_ptr<value_type, __destruct_n&> __h2(__first2, __d);
      if (__comp(*--__last1, *__first1))
      {
        ::cuda::std::__construct_at(__first2, _Ops::__iter_move(__last1));
        __d.template __incr<value_type>();
        ++__first2;
        ::cuda::std::__construct_at(__first2, _Ops::__iter_move(__first1));
      }
      else
      {
        ::cuda::std::__construct_at(__first2, _Ops::__iter_move(__first1));
        __d.template __incr<value_type>();
        ++__first2;
        ::cuda::std::__construct_at(__first2, _Ops::__iter_move(__last1));
      }
      __h2.release();
      return;
  }
  if (__len <= 8)
  {
    ::cuda::std::__insertion_sort_move<_AlgPolicy, _Compare>(__first1, __last1, __first2, __comp);
    return;
  }
  typename iterator_traits<_RandomAccessIterator>::difference_type __l2 = __len / 2;
  _RandomAccessIterator __m                                             = __first1 + __l2;
  ::cuda::std::__stable_sort<_AlgPolicy, _Compare>(__first1, __m, __comp, __l2, __first2, __l2);
  ::cuda::std::__stable_sort<_AlgPolicy, _Compare>(__m, __last1, __comp, __len - __l2, __first2 + __l2, __len - __l2);
  ::cuda::std::__merge_move_construct<_AlgPolicy, _Compare>(__first1, __m, __m, __last1, __first2, __comp);
}

template <class _Tp>
struct __stable_sort_switch
{
  static const unsigned value = 128 * is_trivially_copy_assignable_v<_Tp>;
};

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_CCCL_API void __stable_sort(
  _RandomAccessIterator __first,
  _RandomAccessIterator __last,
  _Compare __comp,
  typename iterator_traits<_RandomAccessIterator>::difference_type __len,
  typename iterator_traits<_RandomAccessIterator>::value_type* __buff,
  ptrdiff_t __buff_size)
{
  using value_type      = typename iterator_traits<_RandomAccessIterator>::value_type;
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  switch (__len)
  {
    case 0:
    case 1:
      return;
    case 2:
      if (__comp(*--__last, *__first))
      {
        _IterOps<_AlgPolicy>::iter_swap(__first, __last);
      }
      return;
  }
  if (__len <= static_cast<difference_type>(__stable_sort_switch<value_type>::value))
  {
    ::cuda::std::__insertion_sort<_AlgPolicy, _Compare>(__first, __last, __comp);
    return;
  }

  typename iterator_traits<_RandomAccessIterator>::difference_type __l2 = __len / 2;
  _RandomAccessIterator __m                                             = __first + __l2;
  if (__len <= __buff_size)
  {
    __destruct_n __d(0);
    unique_ptr<value_type, __destruct_n&> __h2(__buff, __d);
    ::cuda::std::__stable_sort_move<_AlgPolicy, _Compare>(__first, __m, __comp, __l2, __buff);
    __d.__set(__l2, (value_type*) nullptr);
    ::cuda::std::__stable_sort_move<_AlgPolicy, _Compare>(__m, __last, __comp, __len - __l2, __buff + __l2);
    __d.__set(__len, (value_type*) nullptr);
    ::cuda::std::__merge_move_assign<_AlgPolicy, _Compare>(
      __buff, __buff + __l2, __buff + __l2, __buff + __len, __first, __comp);
    return;
  }
  ::cuda::std::__stable_sort<_AlgPolicy, _Compare>(__first, __m, __comp, __l2, __buff, __buff_size);
  ::cuda::std::__stable_sort<_AlgPolicy, _Compare>(__m, __last, __comp, __len - __l2, __buff, __buff_size);
  ::cuda::std::__inplace_merge<_AlgPolicy>(__first, __m, __last, __comp, __l2, __len - __l2, __buff, __buff_size);
}

template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
_CCCL_API void __stable_sort_impl(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare& __comp)
{
  using value_type      = typename iterator_traits<_RandomAccessIterator>::value_type;
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;

  difference_type __len = __last - __first;
  pair<value_type*, ptrdiff_t> __buf(0, 0);
  unique_ptr<value_type, __return_temporary_buffer> __h;
  if (__len > static_cast<difference_type>(__stable_sort_switch<value_type>::value))
  {
    __buf = ::cuda::std::get_temporary_buffer<value_type>(__len);
    __h.reset(__buf.first);
  }

  ::cuda::std::__stable_sort<_AlgPolicy, __comp_ref_type<_Compare>>(
    __first, __last, __comp, __len, __buf.first, __buf.second);
}

template <class _RandomAccessIterator, class _Compare>
_CCCL_API void stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
  ::cuda::std::__stable_sort_impl<_ClassicAlgPolicy>(::cuda::std::move(__first), ::cuda::std::move(__last), __comp);
}

template <class _RandomAccessIterator>
_CCCL_API void stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
  ::cuda::std::stable_sort(__first, __last, __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_STABLE_SORT_H
