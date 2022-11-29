//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_EXTENT_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_EXTENT_H

#ifndef __cuda_std__
#include <__config>
#include <cstddef>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__remove_extent)
template <class _Tp>
struct remove_extent {
  using type _LIBCUDACXX_NODEBUG = __remove_extent(_Tp);
};

template <class _Tp>
using __remove_extent_t = __remove_extent(_Tp);
#else
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_extent
    {typedef _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_extent<_Tp[]>
    {typedef _Tp type;};
template <class _Tp, size_t _Np> struct _LIBCUDACXX_TEMPLATE_VIS remove_extent<_Tp[_Np]>
    {typedef _Tp type;};

template <class _Tp>
using __remove_extent_t = typename remove_extent<_Tp>::type;
#endif // __has_builtin(__remove_extent)

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using remove_extent_t = __remove_extent_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_EXTENT_H
