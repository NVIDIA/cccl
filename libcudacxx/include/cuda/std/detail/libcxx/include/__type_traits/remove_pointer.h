//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_POINTER_H

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

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_REMOVE_POINTER) && !defined(_LIBCUDACXX_USE_REMOVE_POINTER_FALLBACK)
template <class _Tp>
struct remove_pointer {
  using type _LIBCUDACXX_NODEBUG_TYPE = _LIBCUDACXX_REMOVE_POINTER(_Tp);
};

template <class _Tp>
using __remove_pointer_t = _LIBCUDACXX_REMOVE_POINTER(_Tp);

#else
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_pointer                      {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_pointer<_Tp*>                {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_pointer<_Tp* const>          {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_pointer<_Tp* volatile>       {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_pointer<_Tp* const volatile> {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};

template <class _Tp>
using __remove_pointer_t = typename remove_pointer<_Tp>::type;

#endif // defined(_LIBCUDACXX_REMOVE_POINTER) && !defined(_LIBCUDACXX_USE_REMOVE_POINTER_FALLBACK)

#if _CCCL_STD_VER > 2011
template <class _Tp> using remove_pointer_t = __remove_pointer_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_POINTER_H
