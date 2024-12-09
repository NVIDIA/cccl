//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___BIT_BYTESWAP_H
#define _LIBCUDACXX___BIT_BYTESWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/climits>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

class __byteswap_impl
{
  template <class _Tp>
  using __unsigned_type = bool_constant<_CCCL_TRAIT(is_integral, _Tp) && _CCCL_TRAIT(is_unsigned, _Tp)>;

  _CCCL_TEMPLATE(class _Half, class _Full)
  _CCCL_REQUIRES(
    __unsigned_type<_Half>::value _CCCL_AND __unsigned_type<_Full>::value _CCCL_AND(sizeof(_Full) == sizeof(_Half) * 2))
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Full __impl_recursive(_Full __val) noexcept
  {
    return static_cast<_Full>(__impl(static_cast<_Half>(__val >> CHAR_BIT * sizeof(_Half))))
         | (static_cast<_Full>(__impl(static_cast<_Half>(__val))) << CHAR_BIT * sizeof(_Half));
  }

public:
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__unsigned_type<_Tp>::value)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static _CCCL_CONSTEXPR_CXX14 _Tp __impl(_Tp __val) noexcept
  {
    _Tp __result{};
    for (size_t __i{}; __i < sizeof(__val); ++__i)
    {
      __result <<= CHAR_BIT;
      __result |= (__val >> (__i * CHAR_BIT)) & static_cast<_Tp>(UCHAR_MAX);
    }
    return __result;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint16_t __impl(uint16_t __val) noexcept
  {
#if defined(_CCCL_BUILTIN_BSWAP16)
    return _CCCL_BUILTIN_BSWAP16(__val);
#else
    return (__val << CHAR_BIT) | (__val >> CHAR_BIT);
#endif // _CCCL_BUILTIN_BSWAP16
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint32_t __impl(uint32_t __val) noexcept
  {
#if defined(_CCCL_BUILTIN_BSWAP32)
    return _CCCL_BUILTIN_BSWAP32(__val);
#else
    return __impl_recursive<uint16_t>(__val);
#endif // _CCCL_BUILTIN_BSWAP32
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint64_t __impl(uint64_t __val) noexcept
  {
#if defined(_CCCL_BUILTIN_BSWAP64)
    return _CCCL_BUILTIN_BSWAP64(__val);
#else
    return __impl_recursive<uint32_t>(__val);
#endif // _CCCL_BUILTIN_BSWAP64
  }

#if !defined(_LIBCUDACXX_HAS_NO_INT128)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr unsigned __int128 __impl(unsigned __int128 __val) noexcept
  {
#  if defined(_CCCL_BUILTIN_BSWAP128)
    return _CCCL_BUILTIN_BSWAP128(__val);
#  else
    return __impl_recursive<uint64_t>(__val);
#  endif // _CCCL_BUILTIN_BSWAP128
  }
#endif // !_LIBCUDACXX_HAS_NO_INT128
};

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer) _CCCL_AND(sizeof(_Integer) == 1))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Integer byteswap(_Integer __val) noexcept
{
  return __val;
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer) _CCCL_AND(sizeof(_Integer) > 1)
                 _CCCL_AND(has_single_bit(sizeof(_Integer))))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Integer byteswap(_Integer __val) noexcept
{
  return static_cast<_Integer>(__byteswap_impl::__impl(__to_unsigned_like(__val)));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer) _CCCL_AND(sizeof(_Integer) > 1)
                 _CCCL_AND(!has_single_bit(sizeof(_Integer))))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Integer byteswap(_Integer __val) noexcept
{
  return static_cast<_Integer>(__byteswap_impl::__impl(__to_unsigned_like(__val)));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_BYTESWAP_H
