//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if __has_builtin(__remove_volatile)
template <class _Tp>
struct remove_volatile {
  using type _LIBCUDACXX_NODEBUG = __remove_volatile(_Tp);
};

template <class _Tp>
using __remove_volatile_t = __remove_volatile(_Tp);
#else
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_volatile               {typedef _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_volatile<volatile _Tp> {typedef _Tp type;};

template <class _Tp>
using __remove_volatile_t = typename remove_volatile<_Tp>::type;
#endif // __has_builtin(__remove_volatile)

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using remove_volatile_t = __remove_volatile_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_VOLATILE_H
