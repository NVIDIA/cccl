// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_NEXT_H
#define _CUDA_STD___ITERATOR_NEXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/enable_if.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _InputIter)
_CCCL_REQUIRES(__has_input_traversal<_InputIter>)
[[nodiscard]] _CCCL_API constexpr _InputIter
next(_InputIter __x, typename iterator_traits<_InputIter>::difference_type __n = 1)
{
  _CCCL_ASSERT(__n >= 0 || __has_bidirectional_traversal<_InputIter>,
               "Attempt to next(it, n) with negative n on a non-bidirectional iterator");

  ::cuda::std::advance(__x, __n);
  return __x;
}

_CCCL_END_NAMESPACE_CUDA_STD

// [range.iter.op.next]

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES
_CCCL_BEGIN_NAMESPACE_CPO(__next)
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip>)
  [[nodiscard]] _CCCL_API constexpr _Ip operator()(_Ip __x) const
  {
    ++__x;
    return __x;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip>)
  [[nodiscard]] _CCCL_API constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n) const
  {
    ::cuda::std::ranges::advance(__x, __n);
    return __x;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip, class _Sp)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip>&& sentinel_for<_Sp, _Ip>)
  [[nodiscard]] _CCCL_API constexpr _Ip operator()(_Ip __x, _Sp __bound_sentinel) const
  {
    ::cuda::std::ranges::advance(__x, __bound_sentinel);
    return __x;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip, class _Sp)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip>&& sentinel_for<_Sp, _Ip>)
  [[nodiscard]] _CCCL_API constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n, _Sp __bound_sentinel) const
  {
    ::cuda::std::ranges::advance(__x, __n, __bound_sentinel);
    return __x;
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto next = __next::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_NEXT_H
