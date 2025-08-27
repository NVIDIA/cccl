//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_UNIQUE_H
#define _CUDA_STD___ALGORITHM_UNIQUE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/adjacent_find.h>
#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Iter, class _Sent, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::pair<_Iter, _Iter>
__unique(_Iter __first, _Sent __last, _BinaryPredicate&& __pred)
{
  __first = ::cuda::std::adjacent_find(__first, __last, __pred);
  if (__first != __last)
  {
    // ...  a  a  ?  ...
    //      f     i
    _Iter __i = __first;
    for (++__i; ++__i != __last;)
    {
      if (!__pred(*__first, *__i))
      {
        *++__first = _IterOps<_AlgPolicy>::__iter_move(__i);
      }
    }
    ++__first;
    return ::cuda::std::pair<_Iter, _Iter>(::cuda::std::move(__first), ::cuda::std::move(__i));
  }
  return ::cuda::std::pair<_Iter, _Iter>(__first, __first);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _BinaryPredicate>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred)
{
  return ::cuda::std::__unique<_ClassicAlgPolicy>(::cuda::std::move(__first), ::cuda::std::move(__last), __pred).first;
}

template <class _ForwardIterator>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator unique(_ForwardIterator __first, _ForwardIterator __last)
{
  return ::cuda::std::unique(__first, __last, __equal_to{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_UNIQUE_H
