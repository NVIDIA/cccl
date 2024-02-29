//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_LOWER_BOUND_H
#define _LIBCUDACXX___ALGORITHM_LOWER_BOUND_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__algorithm/comp.h"
#include "../__algorithm/half_positive.h"
#include "../__algorithm/iterator_operations.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__iterator/advance.h"
#include "../__iterator/distance.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/is_callable.h"
#include "../__type_traits/remove_reference.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Iter, class _Sent, class _Type, class _Proj, class _Comp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Iter
__lower_bound(_Iter __first, _Sent __last, const _Type& __value, _Comp& __comp, _Proj& __proj)
{
  auto __len = _IterOps<_AlgPolicy>::distance(__first, __last);

  while (__len != 0)
  {
    auto __l2 = _CUDA_VSTD::__half_positive(__len);
    _Iter __m = __first;
    _IterOps<_AlgPolicy>::advance(__m, __l2);
    if (_CUDA_VSTD::__invoke(__comp, _CUDA_VSTD::__invoke(__proj, *__m), __value))
    {
      __first = ++__m;
      __len -= __l2 + 1;
    }
    else
    {
      __len = __l2;
    }
  }
  return __first;
}

template <class _ForwardIterator, class _Tp, class _Compare>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first), const _Tp&>::value, "The comparator has to be callable");
  auto __proj = _CUDA_VSTD::__identity();
  return _CUDA_VSTD::__lower_bound<_ClassicAlgPolicy>(__first, __last, __value, __comp, __proj);
}

template <class _ForwardIterator, class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
  return _CUDA_VSTD::lower_bound(__first, __last, __value, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_LOWER_BOUND_H
