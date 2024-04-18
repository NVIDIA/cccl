// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_REVERSE_ACCESS_H
#define _LIBCUDACXX___ITERATOR_REVERSE_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011

template <class _Tp, size_t _Np>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 reverse_iterator<_Tp*> rbegin(_Tp (&__array)[_Np])
{
  return reverse_iterator<_Tp*>(__array + _Np);
}

template <class _Tp, size_t _Np>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 reverse_iterator<_Tp*> rend(_Tp (&__array)[_Np])
{
  return reverse_iterator<_Tp*>(__array);
}

template <class _Ep>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 reverse_iterator<const _Ep*> rbegin(initializer_list<_Ep> __il)
{
  return reverse_iterator<const _Ep*>(__il.end());
}

template <class _Ep>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 reverse_iterator<const _Ep*> rend(initializer_list<_Ep> __il)
{
  return reverse_iterator<const _Ep*>(__il.begin());
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 auto rbegin(_Cp& __c) -> decltype(__c.rbegin())
{
  return __c.rbegin();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 auto rbegin(const _Cp& __c) -> decltype(__c.rbegin())
{
  return __c.rbegin();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 auto rend(_Cp& __c) -> decltype(__c.rend())
{
  return __c.rend();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 auto rend(const _Cp& __c) -> decltype(__c.rend())
{
  return __c.rend();
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 auto crbegin(const _Cp& __c) -> decltype(_CUDA_VSTD::rbegin(__c))
{
  return _CUDA_VSTD::rbegin(__c);
}

template <class _Cp>
_LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX17 auto crend(const _Cp& __c) -> decltype(_CUDA_VSTD::rend(__c))
{
  return _CUDA_VSTD::rend(__c);
}

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_REVERSE_ACCESS_H
