//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BITMASK_H
#define _CUDA___BIT_BITMASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/shl.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
#  include <cuda/__ptx/instructions/bmsk.h> // IWYU pragma: keep
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_INT128() && _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
extern "C" _CCCL_DEVICE unsigned __nvvm_prmt(unsigned, unsigned, unsigned);
#endif // _CCCL_HAS_INT128() && _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_INT128() && _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()
[[nodiscard]] _CCCL_DEVICE_API inline uint32_t __bitmask_clipped_word(int __start, int __end, int __offset) noexcept
{
  // Share the clipped start between the position and width calculations.
  const auto __clipped_start = ::max(__start, __offset);
  return ::cuda::ptx::bmsk_clamp(__clipped_start - __offset, ::max(0, __end - __clipped_start));
}

[[nodiscard]] _CCCL_DEVICE_API inline __uint128_t __bitmask_128_clipped(int __start, int __width) noexcept
{
  const auto __end = __start + __width;
  const auto __w0  = ::cuda::ptx::bmsk_clamp(__start, __width);
  const auto __w1  = ::cuda::__bitmask_clipped_word(__start, __end, 32);
  const auto __w2  = ::cuda::__bitmask_clipped_word(__start, __end, 64);
  const auto __w3  = ::cuda::__bitmask_clipped_word(__start, __end, 96);
  return (__uint128_t{__w3} << 96) | (__uint128_t{__w2} << 64) | (__uint128_t{__w1} << 32) | __w0;
}

[[nodiscard]] _CCCL_DEVICE_API inline uint32_t __bitmask_clamp_u8x3(uint32_t __packed) noexcept
{
  // The sub.sat.u8x4 path doesn't produce better code on 120f, even though it supports the instruction.
#  if __cccl_ptx_isa >= 940
  NV_IF_TARGET(NV_HAS_FEATURE_SM_107f, ({
                 uint32_t __result;
                 asm("sub.sat.u8x4 %0, %1, %2;" : "=r"(__result) : "r"(__packed), "r"(0x80604020u));
                 return __result;
               }))
#  endif // __cccl_ptx_isa >= 940

  // The low three bytes represent x - 32, x - 64, and x - 96 for 0 <= x <= 128.
  // A borrow can only affect an already-negative byte, whose value remains >= -97.
  const auto __differences = __packed - 0x00604020u;
  const auto __negative    = ::__nvvm_prmt(__differences, uint32_t{0}, 0xba98u);

  // Replicate each byte's sign bit to clear negative coordinates, and discard the fourth byte.
  return __differences & ~__negative & 0x00ffffffu;
}

[[nodiscard]] _CCCL_DEVICE_API inline __uint128_t __bitmask_128(int __start, int __width) noexcept
{
  // Fused scalar clipping is preferable on these architectures.
  NV_IF_TARGET(NV_IS_EXACTLY_SM_90, (return ::cuda::__bitmask_128_clipped(__start, __width);))
  NV_IF_TARGET(NV_IS_EXACTLY_SM_100, (return ::cuda::__bitmask_128_clipped(__start, __width);))
  NV_IF_TARGET(NV_IS_EXACTLY_SM_103, (return ::cuda::__bitmask_128_clipped(__start, __width);))

  const auto __packed_start = ::__byte_perm(static_cast<uint32_t>(__start), 0, 0x0000);
  const auto __packed_width = ::__byte_perm(static_cast<uint32_t>(__width), 0, 0x0000);

  // No byte can overflow because start + width <= 128.
  const auto __starts = ::cuda::__bitmask_clamp_u8x3(__packed_start);
  const auto __ends   = ::cuda::__bitmask_clamp_u8x3(__packed_start + __packed_width);

  // Each endpoint is at least its start, so this subtraction cannot borrow between bytes.
  const auto __widths = __ends - __starts;

  // Higher bytes need not be masked: a nonzero higher start makes this word empty, while a nonzero higher width means
  // this word's mask extends to its end.
  const auto __w0 = ::cuda::ptx::bmsk_clamp(__start, __width);
  const auto __w1 = ::cuda::ptx::bmsk_clamp(__starts, __widths);
  const auto __w2 = ::cuda::ptx::bmsk_clamp(__starts >> 8, __widths >> 8);
  const auto __w3 = ::cuda::ptx::bmsk_clamp(__starts >> 16, __widths >> 16);
  return (__uint128_t{__w3} << 96) | (__uint128_t{__w2} << 64) | (__uint128_t{__w1} << 32) | __w0;
}
#endif // _CCCL_HAS_INT128() && _CCCL_CUDA_COMPILATION() && !_CCCL_TILE_COMPILATION()

template <typename _Tp = uint32_t>
[[nodiscard]] _CCCL_API constexpr _Tp bitmask(int __start, int __width) noexcept
{
  static_assert(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>, "bitmask() requires unsigned integer types");
  [[maybe_unused]] constexpr auto __digits = ::cuda::std::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__width >= 0 && __width <= __digits, "width out of range");
  _CCCL_ASSERT(__start >= 0 && __start <= __digits, "start position out of range");
  _CCCL_ASSERT(__start + __width <= __digits, "start position + width out of range");

#if !_CCCL_TILE_COMPILATION() // error: asm statement is unsupported in tile code
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    // inline ptx prevents constant folding
    if (!_CCCL_IS_CONSTANT_P(__start) || !_CCCL_IS_CONSTANT_P(__width))
    {
      // For uin64_t the fallback path is fine.
      if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
      {
        NV_IF_TARGET(NV_PROVIDES_SM_70, (return ::cuda::ptx::bmsk_clamp(__start, __width);))
      }
#  if _CCCL_HAS_INT128()
      else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
      {
        NV_IF_TARGET(NV_PROVIDES_SM_70, (return ::cuda::__bitmask_128(__start, __width);))
      }
#  endif // _CCCL_HAS_INT128()
    }
  }
#endif // !_CCCL_TILE_COMPILATION()
  return ::cuda::std::shl(static_cast<_Tp>(::cuda::std::shl(_Tp{1}, static_cast<unsigned>(__width)) - 1),
                          static_cast<unsigned>(__start));
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BITMASK_H
