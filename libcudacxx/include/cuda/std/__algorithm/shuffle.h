//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_SHUFFLE_H
#define _CUDA_STD___ALGORITHM_SHUFFLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__random/uniform_int_distribution.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _AlgPolicy, class _RandomAccessIterator, class _Sentinel, class _UniformRandomNumberGenerator>
_CCCL_API inline _RandomAccessIterator
__shuffle(_RandomAccessIterator __first, _Sentinel __last_sentinel, _UniformRandomNumberGenerator&& __g)
{
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  using _Dp             = uniform_int_distribution<ptrdiff_t>;
  using _Pp             = typename _Dp::param_type;

  auto __original_last = _IterOps<_AlgPolicy>::next(__first, __last_sentinel);
  auto __last          = __original_last;
  difference_type __d  = __last - __first;
  if (__d > 1)
  {
    _Dp __uid;
    for (--__last, (void) --__d; __first < __last; ++__first, (void) --__d)
    {
      difference_type __i = __uid(__g, _Pp(0, __d));
      if (__i != difference_type(0))
      {
        _IterOps<_AlgPolicy>::iter_swap(__first, __first + __i);
      }
    }
  }

  return __original_last;
}

template <class _RandomAccessIterator, class _UniformRandomNumberGenerator>
_CCCL_API inline void
shuffle(_RandomAccessIterator __first, _RandomAccessIterator __last, _UniformRandomNumberGenerator&& __g)
{
  (void) ::cuda::std::__shuffle<_ClassicAlgPolicy>(
    ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::forward<_UniformRandomNumberGenerator>(__g));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_SHUFFLE_H
