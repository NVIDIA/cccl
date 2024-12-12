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
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/climits>
#include <cuda/std/cstdint>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

class __byteswap_impl
{
  template <class _Half, class _Full>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Full __impl_recursive(_Full __val) noexcept
  {
    static_assert(sizeof(_Full) == sizeof(_Half) * 2, "Invalid half type passed to __bytswap_impl");

    return static_cast<_Full>(__impl(static_cast<_Half>(__val >> CHAR_BIT * sizeof(_Half))))
         | (static_cast<_Full>(__impl(static_cast<_Half>(__val))) << CHAR_BIT * sizeof(_Half));
  }

#if _CCCL_HAS_CUDA_COMPILER
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static uint32_t __impl_device(uint32_t __val) noexcept
  {
    uint32_t __result;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(__result) : "r"(__val));
    return __result;
  }

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static uint64_t __impl_device(uint64_t __val) noexcept
  {
    uint32_t __hi;
    uint32_t __lo;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(__hi), "=r"(__lo) : "l"(__val));
    asm("prmt.b32 %0, %0, 0, 0x0123;" : "+r"(__hi));
    asm("prmt.b32 %0, %0, 0, 0x0123;" : "+r"(__lo));

    uint64_t __result;
    asm("mov.b64 %0, {%1, %2};" : "=l"(__result) : "r"(__lo), "r"(__hi));

    return __result;
  }
#endif // _CCCL_HAS_CUDA_COMPILER

public:
  template <class _Tp>
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
#else // ^^^ _CCCL_BUILTIN_BSWAP16 ^^^ / vvv !_CCCL_BUILTIN_BSWAP16 vvv
#  if _CCCL_STD_VER >= 2014 && _CCCL_COMPILER(MSVC)
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST, return _byteswap_ushort(__val);)
    }
#  endif // _CCCL_STD_VER >= 2014 && _CCCL_COMPILER(MSVC)
    return (__val << CHAR_BIT) | (__val >> CHAR_BIT);
#endif // !_CCCL_BUILTIN_BSWAP16
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint32_t __impl(uint32_t __val) noexcept
  {
#if defined(_CCCL_BUILTIN_BSWAP32)
    return _CCCL_BUILTIN_BSWAP32(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP32 ^^^ / vvv !_CCCL_BUILTIN_BSWAP32 vvv
#  if _CCCL_STD_VER >= 2014
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
#    if _CCCL_COMPILER(MSVC)
      NV_IF_TARGET(NV_IS_HOST, return _byteswap_ulong(__val);)
#    endif // _CCCL_COMPILER(MSVC)
      NV_IF_TARGET(NV_IS_DEVICE, return __impl_device(__val);)
    }
#  endif // _CCCL_STD_VER >= 2014
    return __impl_recursive<uint16_t>(__val);
#endif // !_CCCL_BUILTIN_BSWAP32
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint64_t __impl(uint64_t __val) noexcept
  {
#if defined(_CCCL_BUILTIN_BSWAP64)
    return _CCCL_BUILTIN_BSWAP64(__val);
#else // ^^^ _CCCL_BUILTIN_BSWAP64 ^^^ / vvv !_CCCL_BUILTIN_BSWAP64 vvv
#  if _CCCL_STD_VER >= 2014
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
#    if _CCCL_COMPILER(MSVC)
      NV_IF_TARGET(NV_IS_HOST, return _byteswap_uint64(__val);)
#    endif // _CCCL_COMPILER(MSVC)
      NV_IF_TARGET(NV_IS_DEVICE, return __impl_device(__val);)
    }
#  endif // _CCCL_STD_VER >= 2014
    return __impl_recursive<uint32_t>(__val);
#endif // !_CCCL_BUILTIN_BSWAP64
  }

#if !defined(_LIBCUDACXX_HAS_NO_INT128)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __uint128_t __impl(__uint128_t __val) noexcept
  {
#  if defined(_CCCL_BUILTIN_BSWAP128)
    return _CCCL_BUILTIN_BSWAP128(__val);
#  else // ^^^ _CCCL_BUILTIN_BSWAP128 ^^^ / vvv !_CCCL_BUILTIN_BSWAP128 vvv
    return __impl_recursive<uint64_t>(__val);
#  endif // !_CCCL_BUILTIN_BSWAP128
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
  return static_cast<_Integer>(__byteswap_impl::__impl(_CUDA_VSTD::__to_unsigned_like(__val)));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer) _CCCL_AND(sizeof(_Integer) > 1)
                 _CCCL_AND(!has_single_bit(sizeof(_Integer))))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Integer byteswap(_Integer __val) noexcept
{
  return static_cast<_Integer>(__byteswap_impl::__impl(_CUDA_VSTD::__to_unsigned_like(__val)));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___BIT_BYTESWAP_H
