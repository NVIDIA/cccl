//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_IS_COMPLEX
#define _LIBCUDACXX___COMPLEX_IS_COMPLEX

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class>
inline constexpr bool __is_complex_v = false;

template <class _Tp>
inline constexpr bool __is_complex_v<complex<_Tp>> = true;

template <class _Tp>
inline constexpr bool __is_complex_v<const complex<_Tp>> = true;

template <class _Tp>
inline constexpr bool __is_complex_v<volatile complex<_Tp>> = true;

template <class _Tp>
inline constexpr bool __is_complex_v<const volatile complex<_Tp>> = true;

template <class _Tp>
using __is_complex = bool_constant<__is_complex_v<_Tp>>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___COMPLEX_IS_COMPLEX
