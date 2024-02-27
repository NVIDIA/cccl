//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ARITHMETIC_H
#define _LIBCUDACXX___CONCEPTS_ARITHMETIC_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/is_arithmetic.h"
#include "../__type_traits/is_floating_point.h"
#include "../__type_traits/is_integral.h"
#include "../__type_traits/is_signed.h"
#include "../__type_traits/is_signed_integer.h"
#include "../__type_traits/is_signed.h"
#include "../__type_traits/is_unsigned_integer.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011

// [concepts.arithmetic], arithmetic concepts

template<class _Tp>
_LIBCUDACXX_CONCEPT integral = _LIBCUDACXX_TRAIT(is_integral, _Tp);

template<class _Tp>
_LIBCUDACXX_CONCEPT signed_integral = integral<_Tp> && _LIBCUDACXX_TRAIT(is_signed, _Tp);

template<class _Tp>
_LIBCUDACXX_CONCEPT unsigned_integral = integral<_Tp> && !signed_integral<_Tp>;

template<class _Tp>
_LIBCUDACXX_CONCEPT floating_point = _LIBCUDACXX_TRAIT(is_floating_point, _Tp);

template <class _Tp>
_LIBCUDACXX_CONCEPT __libcpp_signed_integer = __libcpp_is_signed_integer<_Tp>::value;

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_ARITHMETIC_H
