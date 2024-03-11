//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MIN_ELEMENT_H
#define _LIBCUDACXX___ALGORITHM_MIN_ELEMENT_H

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
#include "../__algorithm/comp_ref_type.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/is_callable.h"
#include "../__utility/move.h"

#ifndef __cuda_std__
#  include <__pragma_push>
#endif // __cuda_std__

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Comp, class _Iter, class _Sent, class _Proj>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Iter
__min_element(_Iter __first, _Sent __last, _Comp __comp, _Proj& __proj)
{
  if (__first == __last)
  {
    return __first;
  }

  _Iter __i = __first;
  while (++__i != __last)
  {
    if (_CUDA_VSTD::__invoke(__comp, _CUDA_VSTD::__invoke(__proj, *__i), _CUDA_VSTD::__invoke(__proj, *__first)))
    {
      __first = __i;
    }
  }

  return __first;
}

template <class _Comp, class _Iter, class _Sent>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Iter
__min_element(_Iter __first, _Sent __last, _Comp __comp)
{
  auto __proj = __identity();
  return _CUDA_VSTD::__min_element<_Comp>(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp, __proj);
}

template <class _ForwardIterator, class _Compare>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  static_assert(__is_cpp17_input_iterator<_ForwardIterator>::value,
                "std::min_element requires a ForwardIterator");
  static_assert(__is_callable<_Compare, decltype(*__first), decltype(*__first)>::value,
                "The comparator has to be callable");

  return _CUDA_VSTD::__min_element<__comp_ref_type<_Compare> >(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __comp);
}

template <class _ForwardIterator>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last)
{
  return _CUDA_VSTD::min_element(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#ifndef __cuda_std__
#  include <__pragma_pop>
#endif // __cuda_std__

#endif // _LIBCUDACXX___ALGORITHM_MIN_ELEMENT_H
