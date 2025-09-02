//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_MIN_ELEMENT_H
#define _CUDA_STD___ALGORITHM_MIN_ELEMENT_H

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
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _Comp, class _Iter, class _Sent, class _Proj>
_CCCL_API constexpr _Iter __min_element(_Iter __first, _Sent __last, _Comp __comp, _Proj& __proj)
{
  if (__first == __last)
  {
    return __first;
  }

  _Iter __i = __first;
  while (++__i != __last)
  {
    if (::cuda::std::__invoke(__comp, ::cuda::std::__invoke(__proj, *__i), ::cuda::std::__invoke(__proj, *__first)))
    {
      __first = __i;
    }
  }

  return __first;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Comp, class _Iter, class _Sent>
_CCCL_API constexpr _Iter __min_element(_Iter __first, _Sent __last, _Comp __comp)
{
  auto __proj = identity();
  return ::cuda::std::__min_element<_Comp>(::cuda::std::move(__first), ::cuda::std::move(__last), __comp, __proj);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Compare>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  static_assert(__is_cpp17_input_iterator<_ForwardIterator>::value, "std::min_element requires a ForwardIterator");
  static_assert(__is_callable<_Compare, decltype(*__first), decltype(*__first)>::value,
                "The comparator has to be callable");

  return ::cuda::std::__min_element<__comp_ref_type<_Compare>>(
    ::cuda::std::move(__first), ::cuda::std::move(__last), __comp);
}

template <class _ForwardIterator>
[[nodiscard]] _CCCL_API constexpr _ForwardIterator min_element(_ForwardIterator __first, _ForwardIterator __last)
{
  return ::cuda::std::min_element(__first, __last, __less{});
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_MIN_ELEMENT_H
