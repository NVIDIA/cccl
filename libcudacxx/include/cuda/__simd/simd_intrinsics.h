//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___SIMD_SIMD_INTRINSICS_H
#define _CUDA___SIMD_SIMD_INTRINSICS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_SIMD_SAT()

#  include <cuda/std/__internal/features.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_u16x2(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#  if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddus2(__lhs, __rhs);))
#  elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.u16x2 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#  endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_u16x2: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_s16x2(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#  if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddss2(__lhs, __rhs);))
#  elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.s16x2 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#  endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_s16x2: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#  if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddus4(__lhs, __rhs);))
#  elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.u8x4 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#  endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_u8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#  if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddss4(__lhs, __rhs);))
#  elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.s8x4 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#  endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_s8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

_CCCL_END_NAMESPACE_CUDA_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_SAT()
#endif // _CUDA___SIMD_SIMD_INTRINSICS_H
