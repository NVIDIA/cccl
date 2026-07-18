//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_ROTATE_H
#define _CUDA_STD___BIT_ROTATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_rotateleft8)
#  define _CCCL_BUILTIN_ROTATELEFT8(...) __builtin_rotateleft8(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateleft16)
#  define _CCCL_BUILTIN_ROTATELEFT16(...) __builtin_rotateleft16(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateleft32)
#  define _CCCL_BUILTIN_ROTATELEFT32(...) __builtin_rotateleft32(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateleft64)
#  define _CCCL_BUILTIN_ROTATELEFT64(...) __builtin_rotateleft64(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateright8)
#  define _CCCL_BUILTIN_ROTATERIGHT8(...) __builtin_rotateright8(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateright16)
#  define _CCCL_BUILTIN_ROTATERIGHT16(...) __builtin_rotateright16(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateright32)
#  define _CCCL_BUILTIN_ROTATERIGHT32(...) __builtin_rotateright32(__VA_ARGS__)
#endif

#if _CCCL_CHECK_BUILTIN(builtin_rotateright64)
#  define _CCCL_BUILTIN_ROTATERIGHT64(...) __builtin_rotateright64(__VA_ARGS__)
#endif

// nvcc doesn't allow clang's rotater left/right builtins
#if _CCCL_CUDA_COMPILER(NVCC)
#  undef _CCCL_BUILTIN_ROTATELEFT8
#  undef _CCCL_BUILTIN_ROTATELEFT16
#  undef _CCCL_BUILTIN_ROTATELEFT32
#  undef _CCCL_BUILTIN_ROTATELEFT64
#  undef _CCCL_BUILTIN_ROTATERIGHT8
#  undef _CCCL_BUILTIN_ROTATERIGHT16
#  undef _CCCL_BUILTIN_ROTATERIGHT32
#  undef _CCCL_BUILTIN_ROTATERIGHT64
#endif // _CCCL_CUDA_COMPILER(NVCC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp rotr(const _Tp __v, const int __cnt) noexcept
{
#if !_CCCL_TILE_COMPILATION() // nvbug6084444: error: "call to non-tile function not supported!"
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    // For _Tp < uint32_t we can repeat the _Tp bits in upper parts of uint32_t and use 32-bit __funnelshift_r to do the
    // rotation.
    if constexpr (sizeof(_Tp) == sizeof(uint8_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __vrep = ::__byte_perm(uint32_t{__v}, uint32_t{__v}, 0x0000);
                     return static_cast<_Tp>(::__funnelshift_r(__vrep, __vrep, __cnt));
                   }))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __vrep = ::__byte_perm(uint32_t{__v}, uint32_t{__v}, 0x1010);
                     return static_cast<_Tp>(::__funnelshift_r(__vrep, __vrep, __cnt));
                   }))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__funnelshift_r(__v, __v, __cnt);))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __hi    = static_cast<uint32_t>(__v >> 32);
                     const auto __lo    = static_cast<uint32_t>(__v);
                     const auto __res_a = ::__funnelshift_r(__lo, __hi, __cnt);
                     const auto __res_b = ::__funnelshift_r(__hi, __lo, __cnt);
                     return (static_cast<uint32_t>(__cnt) % 64 < 32)
                            ? (uint64_t{__res_b} << 32) | __res_a
                            : (uint64_t{__res_a} << 32) | __res_b;
                   }))
    }
#  if _CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
    {
      NV_IF_TARGET(
        NV_IS_DEVICE, ({
          const auto __w0 = static_cast<uint32_t>(__v);
          const auto __w1 = static_cast<uint32_t>(__v >> 32);
          const auto __w2 = static_cast<uint32_t>(__v >> 64);
          const auto __w3 = static_cast<uint32_t>(__v >> 96);

          const auto __res_0 = ::__funnelshift_r(__w0, __w1, __cnt);
          const auto __res_1 = ::__funnelshift_r(__w1, __w2, __cnt);
          const auto __res_2 = ::__funnelshift_r(__w2, __w3, __cnt);
          const auto __res_3 = ::__funnelshift_r(__w3, __w0, __cnt);

          const auto __word_rot = (static_cast<uint32_t>(__cnt) / 32) % 4;
          const auto __is_rot_0 = __word_rot == 0;
          const auto __is_rot_1 = __word_rot == 1;
          const auto __is_rot_2 = __word_rot == 2;

          const auto __out_0 = __is_rot_0 ? __res_0 : (__is_rot_1 ? __res_1 : (__is_rot_2 ? __res_2 : __res_3));
          const auto __out_1 = __is_rot_0 ? __res_1 : (__is_rot_1 ? __res_2 : (__is_rot_2 ? __res_3 : __res_0));
          const auto __out_2 = __is_rot_0 ? __res_2 : (__is_rot_1 ? __res_3 : (__is_rot_2 ? __res_0 : __res_1));
          const auto __out_3 = __is_rot_0 ? __res_3 : (__is_rot_1 ? __res_0 : (__is_rot_2 ? __res_1 : __res_2));

          return (__uint128_t{__out_3} << 96) | (__uint128_t{__out_2} << 64) | (__uint128_t{__out_1} << 32) | __out_0;
        }))
    }
#  endif // _CCCL_HAS_INT128()
  }
#endif // !_CCCL_TILE_COMPILATION()
#if defined(_CCCL_BUILTIN_ROTATERIGHT8)
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
    return _CCCL_BUILTIN_ROTATERIGHT8(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATERIGHT8
#if defined(_CCCL_BUILTIN_ROTATERIGHT16)
  if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    return _CCCL_BUILTIN_ROTATERIGHT16(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATERIGHT16
#if defined(_CCCL_BUILTIN_ROTATERIGHT32)
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return _CCCL_BUILTIN_ROTATERIGHT32(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATERIGHT32
#if defined(_CCCL_BUILTIN_ROTATERIGHT64)
  if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    return _CCCL_BUILTIN_ROTATERIGHT64(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATERIGHT64
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  const auto __cnt_mod    = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return __cnt_mod == 0 ? __v : (__v >> __cnt_mod) | (__v << (__digits - __cnt_mod));
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp rotl(const _Tp __v, const int __cnt) noexcept
{
#if !_CCCL_TILE_COMPILATION() // nvbug6084444: error: "call to non-tile function not supported!"
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    // For _Tp < uint32_t we can repeat the _Tp bits in upper parts of uint32_t and use 32-bit __funnelshift_l to do the
    // rotation.
    if constexpr (sizeof(_Tp) == sizeof(uint8_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __vrep = ::__byte_perm(uint32_t{__v}, uint32_t{__v}, 0x0000);
                     return static_cast<_Tp>(::__funnelshift_l(__vrep, __vrep, __cnt));
                   }))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __vrep = ::__byte_perm(uint32_t{__v}, uint32_t{__v}, 0x1010);
                     return static_cast<_Tp>(::__funnelshift_l(__vrep, __vrep, __cnt));
                   }))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::__funnelshift_l(__v, __v, __cnt);))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __hi    = static_cast<uint32_t>(__v >> 32);
                     const auto __lo    = static_cast<uint32_t>(__v);
                     const auto __res_a = ::__funnelshift_l(__lo, __hi, __cnt);
                     const auto __res_b = ::__funnelshift_l(__hi, __lo, __cnt);
                     return (static_cast<uint32_t>(__cnt) % 64 < 32)
                            ? (uint64_t{__res_a} << 32) | __res_b
                            : (uint64_t{__res_b} << 32) | __res_a;
                   }))
    }
#  if _CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
    {
      NV_IF_TARGET(
        NV_IS_DEVICE, ({
          const auto __w0 = static_cast<uint32_t>(__v);
          const auto __w1 = static_cast<uint32_t>(__v >> 32);
          const auto __w2 = static_cast<uint32_t>(__v >> 64);
          const auto __w3 = static_cast<uint32_t>(__v >> 96);

          const auto __res_0 = ::__funnelshift_l(__w3, __w0, __cnt);
          const auto __res_1 = ::__funnelshift_l(__w0, __w1, __cnt);
          const auto __res_2 = ::__funnelshift_l(__w1, __w2, __cnt);
          const auto __res_3 = ::__funnelshift_l(__w2, __w3, __cnt);

          const auto __word_rot = (static_cast<uint32_t>(__cnt) / 32) % 4;
          const auto __is_rot_0 = __word_rot == 0;
          const auto __is_rot_1 = __word_rot == 1;
          const auto __is_rot_2 = __word_rot == 2;

          const auto __out_0 = __is_rot_0 ? __res_0 : (__is_rot_1 ? __res_3 : (__is_rot_2 ? __res_2 : __res_1));
          const auto __out_1 = __is_rot_0 ? __res_1 : (__is_rot_1 ? __res_0 : (__is_rot_2 ? __res_3 : __res_2));
          const auto __out_2 = __is_rot_0 ? __res_2 : (__is_rot_1 ? __res_1 : (__is_rot_2 ? __res_0 : __res_3));
          const auto __out_3 = __is_rot_0 ? __res_3 : (__is_rot_1 ? __res_2 : (__is_rot_2 ? __res_1 : __res_0));

          return (__uint128_t{__out_3} << 96) | (__uint128_t{__out_2} << 64) | (__uint128_t{__out_1} << 32) | __out_0;
        }))
    }
#  endif // _CCCL_HAS_INT128()
  }
#endif // !_CCCL_TILE_COMPILATION()
#if defined(_CCCL_BUILTIN_ROTATELEFT8)
  if constexpr (sizeof(_Tp) == sizeof(uint8_t))
  {
    return _CCCL_BUILTIN_ROTATELEFT8(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATELEFT8
#if defined(_CCCL_BUILTIN_ROTATELEFT16)
  if constexpr (sizeof(_Tp) == sizeof(uint16_t))
  {
    return _CCCL_BUILTIN_ROTATELEFT16(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATELEFT16
#if defined(_CCCL_BUILTIN_ROTATELEFT32)
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return _CCCL_BUILTIN_ROTATELEFT32(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATELEFT32
#if defined(_CCCL_BUILTIN_ROTATELEFT64)
  if constexpr (sizeof(_Tp) == sizeof(uint64_t))
  {
    return _CCCL_BUILTIN_ROTATELEFT64(__v, __cnt);
  }
#endif // _CCCL_BUILTIN_ROTATELEFT64
  constexpr auto __digits = numeric_limits<_Tp>::digits;
  const auto __cnt_mod    = static_cast<uint32_t>(__cnt) % __digits; // __cnt is always >= 0
  return __cnt_mod == 0 ? __v : (__v << __cnt_mod) | (__v >> (__digits - __cnt_mod));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_ROTATE_H
