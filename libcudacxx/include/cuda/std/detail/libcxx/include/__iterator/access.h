// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ACCESS_H
#define _LIBCUDACXX___ITERATOR_ACCESS_H

#ifndef __cuda_std__
#  include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../cstddef"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, size_t _Np>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Tp* begin(_Tp (&__array)[_Np])
{
  return __array;
}

template <class _Tp, size_t _Np>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Tp* end(_Tp (&__array)[_Np])
{
  return __array + _Np;
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto begin(_Cp& __c) -> decltype(__c.begin())
{
  return __c.begin();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto begin(const _Cp& __c) -> decltype(__c.begin())
{
  return __c.begin();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto end(_Cp& __c) -> decltype(__c.end())
{
  return __c.end();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto end(const _Cp& __c) -> decltype(__c.end())
{
  return __c.end();
}

#if _CCCL_STD_VER > 2011

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto cbegin(const _Cp& __c)
  -> decltype(_CUDA_VSTD::begin(__c))
{
  return _CUDA_VSTD::begin(__c);
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 auto cend(const _Cp& __c)
  -> decltype(_CUDA_VSTD::end(__c))
{
  return _CUDA_VSTD::end(__c);
}

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ACCESS_H
