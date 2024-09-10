//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_COUNTR_H
#define _LIBCUDACXX___BIT_COUNTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/ctz.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Forward decl for recursive use in split word operations
template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countr_zero(_Tp __t) noexcept;

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<sizeof(_Tp) <= sizeof(uint32_t), int>
__countr_zero_dispatch(_Tp __t) noexcept
{
  return __libcpp_ctz(static_cast<uint32_t>(__t));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<sizeof(_Tp) == sizeof(uint64_t), int>
__countr_zero_dispatch(_Tp __t) noexcept
{
  return __libcpp_ctz(static_cast<uint64_t>(__t));
}

template <typename _Tp, int _St = sizeof(_Tp) / sizeof(uint64_t)>
struct __countr_zero_rsh_impl
{
  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __short_circuit(_Tp __t, int __cur, int __count)
  {
    // Stops processing early if non-zero
    return (__cur == numeric_limits<uint64_t>::digits)
           ? __countr_zero_rsh_impl<_Tp, _St - 1>::__count(__t, __cur + __count)
           : __cur + __count;
  }

  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __count(_Tp __t, int __count)
  {
    return __short_circuit(__t >> numeric_limits<uint64_t>::digits, __countr_zero(static_cast<uint64_t>(__t)), __count);
  }
};

template <typename _Tp>
struct __countr_zero_rsh_impl<_Tp, 1>
{
  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __count(_Tp __t, int __count)
  {
    return __count + __countr_zero(static_cast<uint64_t>(__t));
  }
};

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<(sizeof(_Tp) > sizeof(uint64_t)), int>
__countr_zero_dispatch(_Tp __t) noexcept
{
  return __countr_zero_rsh_impl<_Tp>::__count(__t, 0);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countr_zero(_Tp __t) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__countr_zero requires unsigned");

  return __t ? __countr_zero_dispatch(__t) : numeric_limits<_Tp>::digits;
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __countr_one(_Tp __t) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__countr_one requires unsigned");
  return __t != numeric_limits<_Tp>::max() ? __countr_zero(static_cast<_Tp>(~__t)) : numeric_limits<_Tp>::digits;
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int>
countr_zero(_Tp __t) noexcept
{
  return __countr_zero(__t);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int>
countr_one(_Tp __t) noexcept
{
  return __countr_one(__t);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_COUNTR_H
