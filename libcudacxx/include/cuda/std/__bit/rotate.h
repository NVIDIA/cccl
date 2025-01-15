//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_ROTATE_H
#define _LIBCUDACXX___BIT_ROTATE_H

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
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotl(_Tp __t, unsigned int __cnt) noexcept
{
  static_assert(__cccl_is_unsigned_integer<_Tp>::value, "__rotl requires unsigned");
  using __nlt = numeric_limits<_Tp>;

  return ((__cnt % __nlt::digits) == 0)
         ? __t
         : (__t << (__cnt % __nlt::digits)) | (__t >> (__nlt::digits - (__cnt % __nlt::digits)));
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __rotr(_Tp __t, unsigned int __cnt) noexcept
{
  static_assert(__cccl_is_unsigned_integer<_Tp>::value, "__rotr requires unsigned");
  using __nlt = numeric_limits<_Tp>;

  return ((__cnt % __nlt::digits) == 0)
         ? __t
         : (__t >> (__cnt % __nlt::digits)) | (__t << (__nlt::digits - (__cnt % __nlt::digits)));
}

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, _Tp>
rotl(_Tp __t, unsigned int __cnt) noexcept
{
  return __rotl(__t, __cnt);
}

// rotr
template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, _Tp>
rotr(_Tp __t, unsigned int __cnt) noexcept
{
  return __rotr(__t, __cnt);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_ROTATE_H
