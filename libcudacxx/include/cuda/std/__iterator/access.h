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

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __begin
{
struct __fn
{
  template <class _Tp, size_t _Np>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _Tp* operator()(_Tp (&__array)[_Np]) const noexcept
  {
    return __array;
  }

  template <class _Cp>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 auto operator()(_Cp& __c) const
    noexcept(noexcept(__c.begin())) -> decltype(__c.begin())
  {
    return __c.begin();
  }

  template <class _Cp>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 auto operator()(const _Cp& __c) const
    noexcept(noexcept(__c.begin())) -> decltype(__c.begin())
  {
    return __c.begin();
  }
};
} // namespace __begin

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto begin = __begin::__fn{};
} // namespace __cpo

namespace __end
{
struct __fn
{
  template <class _Tp, size_t _Np>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _Tp* operator()(_Tp (&__array)[_Np]) const noexcept
  {
    return __array + _Np;
  }

  template <class _Cp>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 auto operator()(_Cp& __c) const
    noexcept(noexcept(__c.end())) -> decltype(__c.end())
  {
    return __c.end();
  }

  template <class _Cp>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 auto operator()(const _Cp& __c) const
    noexcept(noexcept(__c.end())) -> decltype(__c.end())
  {
    return __c.end();
  }
};
} // namespace __end

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto end = __end::__fn{};
} // namespace __cpo

#if _CCCL_STD_VER >= 2014

namespace __cbegin
{
struct __fn
{
  template <class _Cp>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 auto operator()(const _Cp& __c) const
    noexcept(noexcept(_CUDA_VSTD::begin(__c))) -> decltype(_CUDA_VSTD::begin(__c))
  {
    return _CUDA_VSTD::begin(__c);
  }
};
} // namespace __cbegin

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto cbegin = __cbegin::__fn{};
} // namespace __cpo

namespace __cend
{
struct __fn
{
  template <class _Cp>
  _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 auto operator()(const _Cp& __c) const
    noexcept(noexcept(_CUDA_VSTD::end(__c))) -> decltype(_CUDA_VSTD::end(__c))
  {
    return _CUDA_VSTD::end(__c);
  }
};
} // namespace __cend

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto cend = __cend::__fn{};
} // namespace __cpo

#endif // _CCCL_STD_VER >= 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ACCESS_H
