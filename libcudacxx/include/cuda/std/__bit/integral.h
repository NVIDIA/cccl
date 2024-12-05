//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_INTEGRAL_H
#define _LIBCUDACXX___BIT_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/countl.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr uint32_t __bit_log2(_Tp __t) noexcept
{
  static_assert(__cccl_is_unsigned_integer<_Tp>::value, "__bit_log2 requires unsigned");
  return numeric_limits<_Tp>::digits - 1 - __countl_zero(__t);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<sizeof(_Tp) >= sizeof(uint32_t), _Tp> __ceil2(_Tp __t) noexcept
{
  return _Tp{1} << (numeric_limits<_Tp>::digits - __countl_zero((_Tp) (__t - 1u)));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<sizeof(_Tp) < sizeof(uint32_t), _Tp> __ceil2(_Tp __t) noexcept
{
  return (_Tp) ((1u << ((numeric_limits<_Tp>::digits - __countl_zero((_Tp) (__t - 1u)))
                        + (numeric_limits<unsigned>::digits - numeric_limits<_Tp>::digits)))
                >> (numeric_limits<unsigned>::digits - numeric_limits<_Tp>::digits));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, _Tp> bit_floor(_Tp __t) noexcept
{
  return __t == 0 ? 0 : static_cast<_Tp>(_Tp{1} << __bit_log2(__t));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, _Tp> bit_ceil(_Tp __t) noexcept
{
  return (__t < 2) ? 1 : static_cast<_Tp>(__ceil2(__t));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr enable_if_t<__cccl_is_unsigned_integer<_Tp>::value, int> bit_width(_Tp __t) noexcept
{
  return __t == 0 ? 0 : static_cast<int>(__bit_log2(__t) + 1);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_INTEGRAL_H
