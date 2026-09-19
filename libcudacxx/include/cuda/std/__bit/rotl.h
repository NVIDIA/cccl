//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_ROTL_H
#define _CUDA_STD___BIT_ROTL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_rotateleft8)
#  define _CCCL_BUILTIN_ROTATELEFT8(...) __builtin_rotateleft8(__VA_ARGS__)
#endif // has __builtin_rotateleft8

#if _CCCL_HAS_BUILTIN(__builtin_rotateleft16)
#  define _CCCL_BUILTIN_ROTATELEFT16(...) __builtin_rotateleft16(__VA_ARGS__)
#endif // has __builtin_rotateleft16

#if _CCCL_HAS_BUILTIN(__builtin_rotateleft32)
#  define _CCCL_BUILTIN_ROTATELEFT32(...) __builtin_rotateleft32(__VA_ARGS__)
#endif // has __builtin_rotateleft32

#if _CCCL_HAS_BUILTIN(__builtin_rotateleft64)
#  define _CCCL_BUILTIN_ROTATELEFT64(...) __builtin_rotateleft64(__VA_ARGS__)
#endif // has __builtin_rotateleft64

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __rotl_impl_generic(const _Tp __v, const int __cnt) noexcept
{
  constexpr auto __digits = __num_bits_v<_Tp>;
  const auto __cnt_mod    = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return (__cnt_mod == 0) ? __v : (__v << __cnt_mod) | (__v >> (__digits - __cnt_mod));
}

#if _CCCL_HOST_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp __rotl_impl_host(const _Tp __v, const int __cnt) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
#  if defined(_CCCL_BUILTIN_ROTATELEFT8)
    return _CCCL_BUILTIN_ROTATELEFT8(__v, __cnt);
#  elif _CCCL_COMPILER(MSVC)
    return ::_rotl8(__v, static_cast<unsigned char>(__cnt));
#  else // ^^^ use builtins ^^^ / vvv fallback vvv
    return ::cuda::std::__rotl_impl_generic(__v, __cnt);
#  endif // ^^^ fallback ^^^
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
#  if defined(_CCCL_BUILTIN_ROTATELEFT16)
    return _CCCL_BUILTIN_ROTATELEFT16(__v, __cnt);
#  elif _CCCL_COMPILER(MSVC)
    return ::_rotl16(__v, static_cast<unsigned char>(__cnt));
#  else // ^^^ use builtins ^^^ / vvv fallback vvv
    return ::cuda::std::__rotl_impl_generic(__v, __cnt);
#  endif // ^^^ fallback ^^^
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
#  if defined(_CCCL_BUILTIN_ROTATELEFT32)
    return _CCCL_BUILTIN_ROTATELEFT32(__v, __cnt);
#  elif _CCCL_COMPILER(MSVC)
    return ::_rotl(__v, __cnt);
#  else // ^^^ use builtins ^^^ / vvv fallback vvv
    return ::cuda::std::__rotl_impl_generic(__v, __cnt);
#  endif // ^^^ fallback ^^^
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
#  if defined(_CCCL_BUILTIN_ROTATELEFT64)
    return _CCCL_BUILTIN_ROTATELEFT64(__v, __cnt);
#  elif _CCCL_COMPILER(MSVC)
    return ::_rotl64(__v, __cnt);
#  else // ^^^ use builtins ^^^ / vvv fallback vvv
    return ::cuda::std::__rotl_impl_generic(__v, __cnt);
#  endif // ^^^ fallback ^^^
  }
  else
  {
    return ::cuda::std::__rotl_impl_generic(__v, __cnt);
  }
}
#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __rotl_impl_device(const _Tp __v, const int __cnt) noexcept
{
  // For _Tp < uint32_t we can repeat the _Tp bits in upper parts of uint32_t and use 32-bit __funnelshift_l to do the
  // rotation. For _Tp > uint32_t we can split the type to 32-bit words, use __funneshift_l to produce the result
  // words and reorder them based on the __cnt value.

  // clang-tidy doesn't see the content of NV_IF_TARGET, thus thinks the branches are all empty.
  // NOLINTBEGIN(bugprone-branch-clone)
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
    const auto __vrep = ::__byte_perm(uint32_t{__v}, uint32_t{__v}, 0x0000);
    return static_cast<_Tp>(::__funnelshift_l(__vrep, __vrep, __cnt));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    const auto __vrep = ::__byte_perm(uint32_t{__v}, uint32_t{__v}, 0x1010);
    return static_cast<_Tp>(::__funnelshift_l(__vrep, __vrep, __cnt));
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return ::__funnelshift_l(__v, __v, __cnt);
  }
  else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    const auto __hi    = static_cast<uint32_t>(__v >> 32);
    const auto __lo    = static_cast<uint32_t>(__v);
    const auto __res_a = ::__funnelshift_l(__lo, __hi, __cnt);
    const auto __res_b = ::__funnelshift_l(__hi, __lo, __cnt);
    return (static_cast<uint32_t>(__cnt) % 64 < 32)
           ? (uint64_t{__res_a} << 32) | __res_b
           : (uint64_t{__res_b} << 32) | __res_a;
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
  {
    const auto __w0 = static_cast<uint32_t>(__v);
    const auto __w1 = static_cast<uint32_t>(__v >> 32);
    const auto __w2 = static_cast<uint32_t>(__v >> 64);
    const auto __w3 = static_cast<uint32_t>(__v >> 96);

    const auto __res_0 = ::__funnelshift_l(__w3, __w0, __cnt);
    const auto __res_1 = ::__funnelshift_l(__w0, __w1, __cnt);
    const auto __res_2 = ::__funnelshift_l(__w1, __w2, __cnt);
    const auto __res_3 = ::__funnelshift_l(__w2, __w3, __cnt);

    const auto __cnt_u     = static_cast<uint32_t>(__cnt);
    const auto __word_rot1 = (__cnt_u & 32) != 0;
    const auto __word_rot2 = (__cnt_u & 64) != 0;

    const auto __tmp_0 = __word_rot1 ? __res_3 : __res_0;
    const auto __tmp_1 = __word_rot1 ? __res_0 : __res_1;
    const auto __tmp_2 = __word_rot1 ? __res_1 : __res_2;
    const auto __tmp_3 = __word_rot1 ? __res_2 : __res_3;

    const auto __out_0 = __word_rot2 ? __tmp_2 : __tmp_0;
    const auto __out_1 = __word_rot2 ? __tmp_3 : __tmp_1;
    const auto __out_2 = __word_rot2 ? __tmp_0 : __tmp_2;
    const auto __out_3 = __word_rot2 ? __tmp_1 : __tmp_3;

    return (__uint128_t{__out_3} << 96) | (__uint128_t{__out_2} << 64) | (__uint128_t{__out_1} << 32) | __out_0;
  }
#  endif // _CCCL_HAS_INT128()
  else
  {
    return ::cuda::std::__rotl_impl_generic(__v, __cnt);
  }
  // NOLINTEND(bugprone-branch-clone)
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp rotl(const _Tp __v, const int __cnt) noexcept
{
#if !_CCCL_TILE_COMPILATION() // nvbug6084444: error: "call to non-tile function not supported!"
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, ({ return ::cuda::std::__rotl_impl_host(__v, __cnt); }), ({
                        return ::cuda::std::__rotl_impl_device(__v, __cnt);
                      }))
  }
#endif // !_CCCL_TILE_COMPILATION()

  return ::cuda::std::__rotl_impl_generic(__v, __cnt);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_ROTL_H
