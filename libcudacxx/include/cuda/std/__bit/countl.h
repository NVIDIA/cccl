//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_COUNTL_H
#define _LIBCUDACXX___BIT_COUNTL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/clz.h>
#include <cuda/std/__bit/rotate.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Forward decl for recursive use in split word operations
template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countl_zero(_Tp __t) noexcept;

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<sizeof(_Tp) <= sizeof(uint32_t), int>
__countl_zero_dispatch(_Tp __t) noexcept
{
  return __libcpp_clz(static_cast<uint32_t>(__t)) - (numeric_limits<uint32_t>::digits - numeric_limits<_Tp>::digits);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<sizeof(_Tp) == sizeof(uint64_t), int>
__countl_zero_dispatch(_Tp __t) noexcept
{
  return __libcpp_clz(static_cast<uint64_t>(__t)) - (numeric_limits<uint64_t>::digits - numeric_limits<_Tp>::digits);
}

template <typename _Tp, int _St = sizeof(_Tp) / sizeof(uint64_t)>
struct __countl_zero_rotl_impl
{
  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __short_circuit(_Tp __t, int __cur)
  {
    // This stops processing early if the current word is not empty
    return (__cur == numeric_limits<uint64_t>::digits)
           ? __cur + __countl_zero_rotl_impl<_Tp, _St - 1>::__count(__t)
           : __cur;
  }

  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __countl_iter(_Tp __t)
  {
    // After rotating pass result of clz to another step for processing
    return __short_circuit(__t, __countl_zero(static_cast<uint64_t>(__t)));
  }

  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __count(_Tp __t)
  {
    return __countl_iter(__rotl(__t, numeric_limits<uint64_t>::digits));
  }
};

template <typename _Tp>
struct __countl_zero_rotl_impl<_Tp, 1>
{
  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __count(_Tp __t)
  {
    return __countl_zero(static_cast<uint64_t>(__rotl(__t, numeric_limits<uint64_t>::digits)));
  }
};

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<(sizeof(_Tp) > sizeof(uint64_t)), int>
__countl_zero_dispatch(_Tp __t) noexcept
{
  return __countl_zero_rotl_impl<_Tp>::__count(__t);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countl_zero(_Tp __t) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__countl_zero requires unsigned");
  return __t ? __countl_zero_dispatch(__t) : numeric_limits<_Tp>::digits;
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countl_one(_Tp __t) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__countl_one requires unsigned");
  return __t != numeric_limits<_Tp>::max() ? __countl_zero(static_cast<_Tp>(~__t)) : numeric_limits<_Tp>::digits;
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int>
countl_zero(_Tp __t) noexcept
{
  return __countl_zero(__t);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int>
countl_one(_Tp __t) noexcept
{
  return __countl_one(__t);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_COUNTL_H
