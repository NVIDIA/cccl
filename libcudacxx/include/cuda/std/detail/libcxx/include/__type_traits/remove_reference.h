//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_REFERENCE_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_REFERENCE_H

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

#include "../cstddef"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_REMOVE_REFERENCE_T) && !defined(_LIBCUDACXX_USE_REMOVE_REFERENCE_T_FALLBACK)
template <class _Tp>
struct remove_reference {
  using type _LIBCUDACXX_NODEBUG_TYPE = _LIBCUDACXX_REMOVE_REFERENCE_T(_Tp);
};

template <class _Tp>
using __libcpp_remove_reference_t = _LIBCUDACXX_REMOVE_REFERENCE_T(_Tp);

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_reference        {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_reference<_Tp&>  {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_reference<_Tp&&> {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};

template <class _Tp>
using __libcpp_remove_reference_t = typename remove_reference<_Tp>::type;

#endif // defined(_LIBCUDACXX_REMOVE_REFERENCE_T) && !defined(_LIBCUDACXX_USE_REMOVE_REFERENCE_T_FALLBACK)

#if _CCCL_STD_VER > 2011
template <class _Tp> using remove_reference_t = __libcpp_remove_reference_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_REFERENCE_H
