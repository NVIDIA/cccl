//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_INTEGRAL_H
#define _CUDA_STD___BIT_INTEGRAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/instructions/shl.h>
#  include <cuda/__ptx/instructions/shr.h>
#endif // _CCCL_CUDA_COMPILATION()
#include <cuda/std/__algorithm/max.h> // IWYU pragma: keep
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
_CCCL_API constexpr uint32_t __bit_log2(_Tp __t) noexcept
{
#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
    { // NOLINT(bugprone-branch-clone)
      // There is no SASS CLZ instruction. Instead, clz.b32 generates the equivalent of:
      // bfind.u32 tmp, x;
      // add.s32   ret, -tmp, 31;
      //
      // __bit_log2 functionality is equivalent to bfind.u32 instruction. If we negate the result of __clz and add 31
      // back to it, ptxas optimizes out everything except for bfind.u32, which is what we want.
      //
      // We do all of this to avoid using inline PTX.
      NV_IF_TARGET(NV_IS_DEVICE, ({ return -::__clz(static_cast<int>(__t)) + 31; }))
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    { // NOLINT(bugprone-branch-clone)
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __lo = static_cast<uint32_t>(__t);
                     const auto __hi = static_cast<uint32_t>(__t >> 32);

                     const bool __hi_is_nonzero = (__hi != 0);
                     auto __ret                 = -::__clz((__hi_is_nonzero) ? __hi : __lo) + 31;
                     if (__hi_is_nonzero)
                     {
                       __ret |= 32;
                     }
                     return __ret;
                   }))
    }
#  if _CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
    { // NOLINT(bugprone-branch-clone)
      NV_IF_TARGET(NV_IS_DEVICE, ({
                     const auto __v0 = static_cast<uint32_t>(__t);
                     const auto __v1 = static_cast<uint32_t>(__t >> 32);
                     const auto __v2 = static_cast<uint32_t>(__t >> 64);
                     const auto __v3 = static_cast<uint32_t>(__t >> 96);

                     const bool __hi_is_nonzero   = ((__v2 | __v3) != 0);
                     const auto __lo              = (__hi_is_nonzero) ? __v2 : __v0;
                     const auto __hi              = (__hi_is_nonzero) ? __v3 : __v1;
                     const bool __word_is_nonzero = (__hi != 0);
                     int __ret                    = -::__clz((__word_is_nonzero) ? __hi : __lo) + 31;

                     // As in the 64-bit case, OR in the word offsets to preserve the ~0 sentinel for a zero input.
                     if (__hi_is_nonzero)
                     {
                       __ret |= 64;
                     }
                     if (__word_is_nonzero)
                     {
                       __ret |= 32;
                     }
                     return __ret;
                   }))
    }
#  endif // _CCCL_HAS_INT128()
  }
#endif // !_CCCL_TILE_COMPILATION()
  return numeric_limits<_Tp>::digits - 1 - ::cuda::std::countl_zero(__t);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int bit_width(_Tp __t) noexcept
{
  // if __t == 0, __bit_log2(0) returns 0xFFFFFFFF. Since unsigned overflow is well-defined, the result is -1 + 1 = 0
  auto __ret = ::cuda::std::__bit_log2(__t) + 1;
  _CCCL_ASSUME(__ret <= numeric_limits<_Tp>::digits);
  return static_cast<int>(__ret);
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_ceil(_Tp __t) noexcept
{
  using _Up = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  _CCCL_ASSERT(__t <= numeric_limits<_Tp>::max() / 2, "bit_ceil overflow");
  // if __t == 0, __t - 1 == 0xFFFFFFFF, bit_width(0xFFFFFFFF) returns 32
  auto __width = ::cuda::std::bit_width(static_cast<_Up>(__t) - 1);
#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) <= 8)
    {
      if (!_CCCL_IS_CONSTANT_P(__t)) // inline ptx prevents constant folding
      {
        // CUDA right shift (ptx::shr) returns 0 if the right operand is larger than the number of bits of the type
        // The result is computed as max(1, bit_width(__t - 1)) because it is more efficient than the ternary operator
        NV_IF_TARGET(NV_IS_DEVICE, //
                     (auto __shift = ::cuda::ptx::shl(_Up{1}, __width); // 2^(ceil(log2(__t - 1)))
                      auto __ret   = static_cast<_Tp>(::cuda::std::max(_Up{1}, __shift)); //
                      _CCCL_ASSUME(__ret >= __t);
                      return __ret;))
      }
    }
  }
#endif // !_CCCL_TILE_COMPILATION()
  auto __ret = static_cast<_Tp>(__t <= 1 ? _Up{1} : _Up{1} << __width);
  _CCCL_ASSUME(__ret >= __t);
  return __ret;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_floor(_Tp __t) noexcept
{
  using _Up   = _If<sizeof(_Tp) <= 4, uint32_t, _Tp>;
  auto __log2 = ::cuda::std::__bit_log2(static_cast<_Up>(__t));
  // __bit_log2 returns 0xFFFFFFFF if __t == 0
#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    if constexpr (sizeof(_Tp) <= 8)
    {
      if (!_CCCL_IS_CONSTANT_P(__t)) // inline ptx prevents constant folding
      {
        // CUDA left shift (ptx::shl) returns 0 if the right operand is larger than the number of bits of the type
        // -> the result is 0 if __t == 0
        NV_IF_TARGET(NV_IS_DEVICE, //
                     (auto __ret = static_cast<_Tp>(::cuda::ptx::shl(_Up{1}, __log2)); // 2^(log2(t))
                      _CCCL_ASSUME(__ret >= __t / 2 && __ret <= __t);
                      return __ret;))
      }
    }
  }
#endif // !_CCCL_TILE_COMPILATION()
  auto __ret = static_cast<_Tp>(__t == 0 ? _Up{0} : _Up{1} << __log2);
  _CCCL_ASSUME(__ret >= __t / 2 && __ret <= __t);
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_INTEGRAL_H
