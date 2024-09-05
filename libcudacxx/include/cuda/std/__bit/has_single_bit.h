//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_HAS_SINGLE_BIT_H
#define _LIBCUDACXX___BIT_HAS_SINGLE_BIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __has_single_bit(_Tp __t) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__has_single_bit requires unsigned");
  return __t != 0 && (((__t & (__t - 1)) == 0));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, bool>
has_single_bit(_Tp __t) noexcept
{
  return __has_single_bit(__t);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_HAS_SINGLE_BIT_H
