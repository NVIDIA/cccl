//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_ARITHMETIC_INTEGRAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_ARITHMETIC_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_cv.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __is_arithmetic_integral is a trait that tests whether a type is an integral type intended for arithmetic.
// In contrast to is_integral, __is_arithmetic_integral excludes bool and character types.

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v = false;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<signed char> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<unsigned char> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<short> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<unsigned short> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<int> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<unsigned int> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<long> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<unsigned long> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<long long> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<unsigned long long> = true;

#if _CCCL_HAS_INT128()
template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<__int128_t> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_impl_v<__uint128_t> = true;
#endif // _CCCL_HAS_INT128()

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_arithmetic_integral_v = __is_arithmetic_integral_impl_v<remove_cv_t<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_ARITHMETIC_INTEGRAL_H
