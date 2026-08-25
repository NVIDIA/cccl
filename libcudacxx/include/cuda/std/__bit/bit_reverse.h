//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_REVERSE_H
#define _CUDA_STD___BIT_BIT_REVERSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_bitreverse8)
#  define _CCCL_BUILTIN_BITREVERSE8(...) __builtin_bitreverse8(__VA_ARGS__)
#endif // has __builtin_bitreverse8

#if _CCCL_HAS_BUILTIN(__builtin_bitreverse16)
#  define _CCCL_BUILTIN_BITREVERSE16(...) __builtin_bitreverse16(__VA_ARGS__)
#endif // has __builtin_bitreverse16

#if _CCCL_HAS_BUILTIN(__builtin_bitreverse32)
#  define _CCCL_BUILTIN_BITREVERSE32(...) __builtin_bitreverse32(__VA_ARGS__)
#endif // has __builtin_bitreverse32

#if _CCCL_HAS_BUILTIN(__builtin_bitreverse64)
#  define _CCCL_BUILTIN_BITREVERSE64(...) __builtin_bitreverse64(__VA_ARGS__)
#endif // has __builtin_bitreverse64

#if _CCCL_HAS_BUILTIN(__builtin_bitreverse128)
#  define _CCCL_BUILTIN_BITREVERSE128(...) __builtin_bitreverse128(__VA_ARGS__)
#endif // has __builtin_bitreverse128

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_reverse_impl_generic(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
    __v = ((__v >> 1) & 0x55) | ((__v & 0x55) << 1);
    __v = ((__v >> 2) & 0x33) | ((__v & 0x33) << 2);
    return (__v >> 4) | (__v << 4);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    __v = ((__v >> 1) & 0x5555) | ((__v & 0x5555) << 1);
    __v = ((__v >> 2) & 0x3333) | ((__v & 0x3333) << 2);
    __v = ((__v >> 4) & 0x0F0F) | ((__v & 0x0F0F) << 4);
    return (__v >> 8) | (__v << 8);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    __v = ((__v >> 1) & 0x55555555) | ((__v & 0x55555555) << 1);
    __v = ((__v >> 2) & 0x33333333) | ((__v & 0x33333333) << 2);
    __v = ((__v >> 4) & 0x0F0F0F0F) | ((__v & 0x0F0F0F0F) << 4);
    __v = ((__v >> 8) & 0x00FF00FF) | ((__v & 0x00FF00FF) << 8);
    return (__v >> 16) | (__v << 16);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    __v = ((__v >> 1) & 0x5555555555555555ull) | ((__v & 0x5555555555555555ull) << 1);
    __v = ((__v >> 2) & 0x3333333333333333ull) | ((__v & 0x3333333333333333ull) << 2);
    __v = ((__v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((__v & 0x0F0F0F0F0F0F0F0Full) << 4);
    __v = ((__v >> 8) & 0x00FF00FF00FF00FFull) | ((__v & 0x00FF00FF00FF00FFull) << 8);
    __v = ((__v >> 16) & 0x0000FFFF0000FFFFull) | ((__v & 0x0000FFFF0000FFFFull) << 16);
    return (__v >> 32) | (__v << 32);
  }
#if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    constexpr auto __c1 = __uint128_t{0x5555555555555555ull} << 64 | 0x5555555555555555ull;
    constexpr auto __c2 = __uint128_t{0x3333333333333333ull} << 64 | 0x3333333333333333ull;
    constexpr auto __c3 = __uint128_t{0x0F0F0F0F0F0F0F0Full} << 64 | 0x0F0F0F0F0F0F0F0Full;
    constexpr auto __c4 = __uint128_t{0x00FF00FF00FF00FFull} << 64 | 0x00FF00FF00FF00FFull;
    constexpr auto __c5 = __uint128_t{0x0000FFFF0000FFFFull} << 64 | 0x0000FFFF0000FFFFull;
    constexpr auto __c6 = __uint128_t{0x00000000FFFFFFFFull} << 64 | 0x00000000FFFFFFFFull;
    __v                 = ((__v >> 1) & __c1) | ((__v & __c1) << 1);
    __v                 = ((__v >> 2) & __c2) | ((__v & __c2) << 2);
    __v                 = ((__v >> 4) & __c3) | ((__v & __c3) << 4);
    __v                 = ((__v >> 8) & __c4) | ((__v & __c4) << 8);
    __v                 = ((__v >> 16) & __c5) | ((__v & __c5) << 16);
    __v                 = ((__v >> 32) & __c6) | ((__v & __c6) << 32);
    return (__v >> 64) | (__v << 64);
  }
#endif // _CCCL_HAS_INT128()
  else
  {
    static_assert(__always_false_v<_Tp>, "Unsupported integer type");
  }
}

#if _CCCL_HOST_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __bit_reverse_impl_host(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
#  if defined(_CCCL_BUILTIN_BITREVERSE8)
    return _CCCL_BUILTIN_BITREVERSE8(__v);
#  else // ^^^ _CCCL_BUILTIN_BITREVERSE8 ^^^ / vvv !_CCCL_BUILTIN_BITREVERSE8 vvv
    return ::cuda::std::__bit_reverse_impl_generic(__v);
#  endif // ^^^ !_CCCL_BUILTIN_BITREVERSE8 ^^^
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
#  if defined(_CCCL_BUILTIN_BITREVERSE16)
    return _CCCL_BUILTIN_BITREVERSE16(__v);
#  else // ^^^ _CCCL_BUILTIN_BITREVERSE16 ^^^ / vvv !_CCCL_BUILTIN_BITREVERSE16 vvv
    return ::cuda::std::__bit_reverse_impl_generic(__v);
#  endif // ^^^ !_CCCL_BUILTIN_BITREVERSE16 ^^^
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
#  if defined(_CCCL_BUILTIN_BITREVERSE32)
    return _CCCL_BUILTIN_BITREVERSE32(__v);
#  else // ^^^ _CCCL_BUILTIN_BITREVERSE32 ^^^ / vvv !_CCCL_BUILTIN_BITREVERSE32 vvv
    return ::cuda::std::__bit_reverse_impl_generic(__v);
#  endif // ^^^ !_CCCL_BUILTIN_BITREVERSE32 ^^^
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
#  if defined(_CCCL_BUILTIN_BITREVERSE64)
    return _CCCL_BUILTIN_BITREVERSE64(__v);
#  else // ^^^ _CCCL_BUILTIN_BITREVERSE64 ^^^ / vvv !_CCCL_BUILTIN_BITREVERSE64 vvv
    return ::cuda::std::__bit_reverse_impl_generic(__v);
#  endif // ^^^ !_CCCL_BUILTIN_BITREVERSE64 ^^^
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
#    if defined(_CCCL_BUILTIN_BITREVERSE128)
    return _CCCL_BUILTIN_BITREVERSE128(__v);
#    elif defined(_CCCL_BUILTIN_BITREVERSE64)
    return (__uint128_t{_CCCL_BUILTIN_BITREVERSE64(static_cast<uint64_t>(__v))} << 64)
         | _CCCL_BUILTIN_BITREVERSE64(static_cast<uint64_t>(__v >> 64));
#    else // ^^^ _CCCL_BUILTIN_BITREVERSE64 ^^^ / vvv !_CCCL_BUILTIN_BITREVERSE64 vvv
    return ::cuda::std::__bit_reverse_impl_generic(__v);
#    endif // ^^^ !_CCCL_BUILTIN_BITREVERSE64 ^^^
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::std::__bit_reverse_impl_generic(__v);
  }
}
#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_DEVICE_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API constexpr _Tp __bit_reverse_impl_device(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
    return ::__brev(uint32_t{__v} << 24);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    return ::__brev(uint32_t{__v} << 16);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return ::__brev(__v);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    return ::__brevll(__v);
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    return (__uint128_t{::__brevll(static_cast<uint64_t>(__v))} << 64) | ::__brevll(static_cast<uint64_t>(__v >> 64));
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::std::__bit_reverse_impl_generic(__v);
  }
}
#endif // _CCCL_DEVICE_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_reverse(_Tp __v) noexcept
{
#if !_CCCL_TILE_COMPILATION() // nvbug6085411: error: "call to non-tile function not supported!"
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, ({ return ::cuda::std::__bit_reverse_impl_host(__v); }), ({
                        return ::cuda::std::__bit_reverse_impl_device(__v);
                      }))
  }
#endif // !_CCCL_TILE_COMPILATION()
  return ::cuda::std::__bit_reverse_impl_generic(__v);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_REVERSE_H
