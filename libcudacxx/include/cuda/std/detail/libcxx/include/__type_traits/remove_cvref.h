//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_REMOVE_CVREF_H
#define _LIBCUDACXX___TYPE_TRAITS_REMOVE_CVREF_H

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

#include "../__type_traits/is_same.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/remove_reference.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_REMOVE_CVREF) && !defined(_LIBCUDACXX_USE_REMOVE_CVREF_FALLBACK)

template <class _Tp>
using __remove_cvref_t _LIBCUDACXX_NODEBUG_TYPE = _LIBCUDACXX_REMOVE_CVREF(_Tp);

#else

template <class _Tp>
using __remove_cvref_t _LIBCUDACXX_NODEBUG_TYPE = __remove_cv_t<__libcpp_remove_reference_t<_Tp> >;

#endif // defined(_LIBCUDACXX_REMOVE_CVREF) && !defined(_LIBCUDACXX_USE_REMOVE_CVREF_FALLBACK)

template <class _Tp, class _Up>
struct __is_same_uncvref : _IsSame<__remove_cvref_t<_Tp>, __remove_cvref_t<_Up> > {};

#if _CCCL_STD_VER > 2011
template <class _Tp>
struct remove_cvref {
    using type _LIBCUDACXX_NODEBUG_TYPE = __remove_cvref_t<_Tp>;
};

template <class _Tp> using remove_cvref_t = __remove_cvref_t<_Tp>;
#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_REMOVE_CVREF_H
