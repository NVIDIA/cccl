//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp> struct __libcpp_is_unsigned_integer : public false_type {};
template <> struct __libcpp_is_unsigned_integer<unsigned char>      : public true_type {};
template <> struct __libcpp_is_unsigned_integer<unsigned short>     : public true_type {};
template <> struct __libcpp_is_unsigned_integer<unsigned int>       : public true_type {};
template <> struct __libcpp_is_unsigned_integer<unsigned long>      : public true_type {};
template <> struct __libcpp_is_unsigned_integer<unsigned long long> : public true_type {};
#ifndef _LIBCUDACXX_HAS_NO_INT128
template <> struct __libcpp_is_unsigned_integer<__uint128_t>        : public true_type {};
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_INTEGER_H
