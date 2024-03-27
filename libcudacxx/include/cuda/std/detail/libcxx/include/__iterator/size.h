// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_SIZE_H
#define _LIBCUDACXX___ITERATOR_SIZE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__type_traits/common_type.h"
#include "../__type_traits/make_signed.h"
#include "../cstddef"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011
template <class _Cont>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr auto size(const _Cont& __c)
noexcept(noexcept(__c.size()))
-> decltype        (__c.size())
{ return            __c.size(); }

template <class _Tp, size_t _Sz>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr size_t size(const _Tp (&)[_Sz]) noexcept { return _Sz; }
#endif // _CCCL_STD_VER > 2011

#if _CCCL_STD_VER > 2017
template <class _Cont>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr auto ssize(const _Cont& __c)
noexcept(noexcept(static_cast<common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>>(__c.size())))
->                              common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>
{ return            static_cast<common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>>(__c.size()); }

// GCC complains about the implicit conversion from ptrdiff_t to size_t in
// the array bound.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wsign-conversion")
template <class _Tp, ptrdiff_t _Sz>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr ptrdiff_t ssize(const _Tp (&)[_Sz]) noexcept { return _Sz; }
_CCCL_DIAG_POP
#endif  // _CCCL_STD_VER > 2017

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_SIZE_H
