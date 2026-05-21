//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_SPECIALIZATIONS_SIMD_INTRINSICS_H
#define _CUDA_STD___SIMD_SPECIALIZATIONS_SIMD_INTRINSICS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

[[nodiscard]] _CCCL_DEVICE_API inline uint32_t
__vadd_u16x2([[maybe_unused]] const uint32_t __lhs, [[maybe_unused]] const uint32_t __rhs) noexcept
{
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (return ::__vadd2(__lhs, __rhs);), //
               (_CCCL_VERIFY(false, "cuda::std::simd::__vadd_u16x2: Unsupported architecture"); return uint32_t{};));
}

[[nodiscard]] _CCCL_DEVICE_API inline uint32_t
__vadd_s16x2([[maybe_unused]] const uint32_t __lhs, [[maybe_unused]] const uint32_t __rhs) noexcept
{
  // prevent MSVC warning
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               ({
                 uint32_t __result{};
                 asm("add.s16x2 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }),
               (_CCCL_VERIFY(false, "cuda::std::simd::__vadd_s16x2: Unsupported architecture"); return uint32_t{};));
}

#  if _CCCL_HAS_SIMD_8BIT()

[[nodiscard]] _CCCL_DEVICE_API inline uint32_t
__vadd_u8x4([[maybe_unused]] const uint32_t __lhs, [[maybe_unused]] const uint32_t __rhs) noexcept
{
#    if _CCCL_HAS_SIMD_8BIT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
               (return ::__vadd4(__lhs, __rhs);), //
               (_CCCL_VERIFY(false, "cuda::std::simd::__vadd_u8x4: Unsupported architecture"); return uint32_t{};));
#    else // ^^^ _CCCL_HAS_SIMD_8BIT_INTRINSICS() ^^^ / vvv !_CCCL_HAS_SIMD_8BIT_INTRINSICS() vvv
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
               ({
                 uint32_t __result{};
                 asm("add.u8x4 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }),
               (_CCCL_VERIFY(false, "cuda::std::simd::__vadd_u8x4: Unsupported architecture"); return uint32_t{};));
#    endif // _CCCL_HAS_SIMD_8BIT()
}

[[nodiscard]] _CCCL_DEVICE_API inline uint32_t
__vadd_s8x4([[maybe_unused]] const uint32_t __lhs, [[maybe_unused]] const uint32_t __rhs) noexcept
{
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
               ({
                 uint32_t __result{};
                 asm("add.s8x4 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }),
               (_CCCL_VERIFY(false, "cuda::std::simd::__vadd_s8x4: Unsupported architecture"); return uint32_t{};));
}

#  endif // _CCCL_HAS_SIMD_8BIT()

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#  include <cuda/std/__cccl/epilogue.h>
#endif // _CCCL_CUDA_COMPILATION()
#endif // _CUDA_STD___SIMD_SPECIALIZATIONS_SIMD_INTRINSICS_H
