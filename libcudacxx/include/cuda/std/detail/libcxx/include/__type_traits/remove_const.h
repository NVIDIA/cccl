//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_CONST_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_CONST_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_REMOVE_CONST) && !defined(_LIBCUDACXX_USE_REMOVE_CONST_FALLBACK)
template <class _Tp>
struct remove_const {
  using type _LIBCUDACXX_NODEBUG_TYPE = _LIBCUDACXX_REMOVE_CONST(_Tp);
};

template <class _Tp>
using __remove_const_t = _LIBCUDACXX_REMOVE_CONST(_Tp);

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_const            {typedef _Tp type;};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS remove_const<const _Tp> {typedef _Tp type;};

template <class _Tp>
using __remove_const_t = typename remove_const<_Tp>::type;

#endif // defined(_LIBCUDACXX_REMOVE_CONST) && !defined(_LIBCUDACXX_USE_REMOVE_CONST_FALLBACK)

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using remove_const_t = __remove_const_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_CONST_H
