//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_FILL_N_H
#define _LIBCUDACXX___ALGORITHM_FILL_N_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#include "../__iterator/iterator_traits.h"
#include "../__utility/convert_to_integral.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_HEADER)
#  pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC
_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _OutputIterator, class _Size, class _Tp>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _OutputIterator
__fill_n(_OutputIterator __first, _Size __n, const _Tp& __value_)
{
  for (; __n > 0; ++__first, (void) --__n)
  {
    *__first = __value_;
  }
  return __first;
}

template <class _OutputIterator, class _Size, class _Tp>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _OutputIterator
fill_n(_OutputIterator __first, _Size __n, const _Tp& __value_)
{
  return _CUDA_VSTD::__fill_n(__first, __convert_to_integral(__n), __value_);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_FILL_N_H
