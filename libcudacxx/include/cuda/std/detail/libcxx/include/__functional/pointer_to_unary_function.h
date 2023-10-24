// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_POINTER_TO_UNARY_FUNCTION_H
#define _LIBCUDACXX___FUNCTIONAL_POINTER_TO_UNARY_FUNCTION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/unary_function.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

template <class _Arg, class _Result>
class _LIBCUDACXX_TEMPLATE_VIS _LIBCUDACXX_DEPRECATED_IN_CXX11 pointer_to_unary_function
    : public __unary_function<_Arg, _Result>
{
    _Result (*__f_)(_Arg);
public:
    _LIBCUDACXX_HIDE_FROM_ABI explicit pointer_to_unary_function(_Result (*__f)(_Arg))
        : __f_(__f) {}
    _LIBCUDACXX_HIDE_FROM_ABI _Result operator()(_Arg __x) const
        {return __f_(__x);}
};

template <class _Arg, class _Result>
_LIBCUDACXX_DEPRECATED_IN_CXX11 _LIBCUDACXX_HIDE_FROM_ABI
pointer_to_unary_function<_Arg,_Result>
ptr_fun(_Result (*__f)(_Arg))
    {return pointer_to_unary_function<_Arg,_Result>(__f);}

#endif // _LIBCUDACXX_STD_VER <= 14 || defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_BINDERS)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_POINTER_TO_UNARY_FUNCTION_H
