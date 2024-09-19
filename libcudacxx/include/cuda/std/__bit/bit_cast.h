//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_BIT_CAST_H
#define _LIBCUDACXX___BIT_BIT_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_default_constructible.h>
#include <cuda/std/detail/libcxx/include/cstring>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_BIT_CAST)
#  define _LIBCUDACXX_CONSTEXPR_BIT_CAST constexpr
#else // ^^^ _LIBCUDACXX_BIT_CAST ^^^ / vvv !_LIBCUDACXX_BIT_CAST vvv
#  define _LIBCUDACXX_CONSTEXPR_BIT_CAST
#  if defined(_CCCL_COMPILER_GCC) && __GNUC__ >= 8
// GCC starting with GCC8 warns about our extended floating point types having protected data members
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wclass-memaccess")
#  endif // _CCCL_COMPILER_GCC >= 8
#endif // !_LIBCUDACXX_BIT_CAST

template <
  class _To,
  class _From,
  __enable_if_t<(sizeof(_To) == sizeof(_From)), int>                                                                = 0,
  __enable_if_t<_CCCL_TRAIT(is_trivially_copyable, _To) || _CCCL_TRAIT(__is_extended_floating_point, _To), int>     = 0,
  __enable_if_t<_CCCL_TRAIT(is_trivially_copyable, _From) || _CCCL_TRAIT(__is_extended_floating_point, _From), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST _To bit_cast(const _From& __from) noexcept
{
#if defined(_LIBCUDACXX_BIT_CAST)
  return _LIBCUDACXX_BIT_CAST(_To, __from);
#else // ^^^ _LIBCUDACXX_BIT_CAST ^^^ / vvv !_LIBCUDACXX_BIT_CAST vvv
  static_assert(_CCCL_TRAIT(is_trivially_default_constructible, _To),
                "The compiler does not support __builtin_bit_cast, so bit_cast additionally requires the destination "
                "type to be trivially constructible");
  _To __temp;
  _CUDA_VSTD::memcpy(&__temp, &__from, sizeof(_To));
  return __temp;
#endif // !_LIBCUDACXX_BIT_CAST
}

#if !defined(_LIBCUDACXX_BIT_CAST)
#  if defined(_CCCL_COMPILER_GCC) && __GNUC__ >= 8
_CCCL_DIAG_POP
#  endif // _CCCL_COMPILER_GCC >= 8
#endif // !_LIBCUDACXX_BIT_CAST

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_BIT_CAST_H
