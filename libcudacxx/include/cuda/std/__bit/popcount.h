//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_POPCOUNT_H
#define _LIBCUDACXX___BIT_POPCOUNT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/popc.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<sizeof(_Tp) <= sizeof(uint32_t), int>
__popcount_dispatch(_Tp __t) noexcept
{
  return __libcpp_popc(static_cast<uint32_t>(__t));
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<sizeof(_Tp) == sizeof(uint64_t), int>
__popcount_dispatch(_Tp __t) noexcept
{
  return __libcpp_popc(static_cast<uint64_t>(__t));
}

template <typename _Tp, int _St = sizeof(_Tp) / sizeof(uint64_t)>
struct __popcount_rsh_impl
{
  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __count(_Tp __t)
  {
    return __popcount_rsh_impl<_Tp, _St - 1>::__count(__t >> numeric_limits<uint64_t>::digits)
         + __libcpp_popc(static_cast<uint64_t>(__t));
  }
};

template <typename _Tp>
struct __popcount_rsh_impl<_Tp, 1>
{
  static _LIBCUDACXX_HIDE_FROM_ABI constexpr int __count(_Tp __t)
  {
    return __libcpp_popc(static_cast<uint64_t>(__t));
  }
};

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<(sizeof(_Tp) > sizeof(uint64_t)), int>
__popcount_dispatch(_Tp __t) noexcept
{
  return __popcount_rsh_impl<_Tp>::__count(__t);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __popcount(_Tp __t) noexcept
{
  static_assert(__libcpp_is_unsigned_integer<_Tp>::value, "__libcpp_popcount requires unsigned");

  return __popcount_dispatch(__t);
}

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __enable_if_t<__libcpp_is_unsigned_integer<_Tp>::value, int>
popcount(_Tp __t) noexcept
{
  return __popcount(__t);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_POPCOUNT_H
