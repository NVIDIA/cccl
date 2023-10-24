//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_CONVERT_TO_INTEGRAL_H
#define _LIBCUDACXX___UTILITY_CONVERT_TO_INTEGRAL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_enum.h"
#include "../__type_traits/is_floating_point.h"
#include "../__type_traits/underlying_type.h"

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI constexpr
int __convert_to_integral(int __val) { return __val; }

_LIBCUDACXX_HIDE_FROM_ABI constexpr
unsigned __convert_to_integral(unsigned __val) { return __val; }

_LIBCUDACXX_HIDE_FROM_ABI constexpr
long __convert_to_integral(long __val) { return __val; }

_LIBCUDACXX_HIDE_FROM_ABI constexpr
unsigned long __convert_to_integral(unsigned long __val) { return __val; }

_LIBCUDACXX_HIDE_FROM_ABI constexpr
long long __convert_to_integral(long long __val) { return __val; }

_LIBCUDACXX_HIDE_FROM_ABI constexpr
unsigned long long __convert_to_integral(unsigned long long __val) {return __val; }

template<typename _Fp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr
__enable_if_t<is_floating_point<_Fp>::value, long long>
 __convert_to_integral(_Fp __val) { return __val; }

#ifndef _LIBCUDACXX_HAS_NO_INT128
_LIBCUDACXX_HIDE_FROM_ABI constexpr
__int128_t __convert_to_integral(__int128_t __val) { return __val; }

_LIBCUDACXX_HIDE_FROM_ABI constexpr
__uint128_t __convert_to_integral(__uint128_t __val) { return __val; }
#endif

template <class _Tp, bool = is_enum<_Tp>::value>
struct __sfinae_underlying_type
{
    typedef typename underlying_type<_Tp>::type type;
    typedef decltype(((type)1) + 0) __promoted_type;
};

template <class _Tp>
struct __sfinae_underlying_type<_Tp, false> {};

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr
typename __sfinae_underlying_type<_Tp>::__promoted_type
__convert_to_integral(_Tp __val) { return __val; }

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_CONVERT_TO_INTEGRAL_H
