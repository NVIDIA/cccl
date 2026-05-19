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

#if _CCCL_HAS_SIMD_SAT() || _CCCL_HAS_SIMD_VABSDIFF() || _CCCL_HAS_SIMD_IDOT()

#  include <cuda/std/__internal/features.h>
#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_SIMD

#  if _CCCL_HAS_SIMD_SAT()

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_u16x2(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#    if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddus2(__lhs, __rhs);))
#    elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.u16x2 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_u16x2: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_s16x2(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#    if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddss2(__lhs, __rhs);))
#    elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.s16x2 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_s16x2: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#    if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddus4(__lhs, __rhs);))
#    elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.u8x4 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_u8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vadd_sat_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs, [[maybe_unused]] const ::cuda::std::uint32_t __rhs) noexcept
{
#    if _CCCL_HAS_SIMD_SAT_INTRINSICS()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, (return ::__vaddss4(__lhs, __rhs);))
#    elif _CCCL_HAS_SIMD_SAT_PTX()
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f, ({
                 ::cuda::std::uint32_t __result{};
                 asm("add.sat.s8x4 %0, %1, %2;" : "=r"(__result) : "r"(__lhs), "r"(__rhs));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_SAT_INTRINSICS() || _CCCL_HAS_SIMD_SAT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__vadd_sat_s8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

#  endif // _CCCL_HAS_SIMD_SAT()

#  if _CCCL_HAS_SIMD_VABSDIFF()

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vabsdiff_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __c) noexcept
{
  NV_IF_TARGET(NV_IS_DEVICE, ({
                 ::cuda::std::uint32_t __result{};
                 asm("vabsdiff4.u32.u32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__c));
                 return __result;
               }))
  _CCCL_VERIFY(false, "cuda::__simd::__vabsdiff_u8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __vabsdiff_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __c) noexcept
{
  NV_IF_TARGET(NV_IS_DEVICE, ({
                 ::cuda::std::uint32_t __result{};
                 asm("vabsdiff4.u32.s32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__c));
                 return __result;
               }))
  _CCCL_VERIFY(false, "cuda::__simd::__vabsdiff_s8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

#  endif // _CCCL_HAS_SIMD_VABSDIFF()

#  if _CCCL_HAS_SIMD_IDOT()

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __dp4a_u8x4_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_INTRINSICS()
  NV_IF_TARGET(NV_PROVIDES_SM_61, (return ::__dp4a(__lhs, __rhs, __acc);))
#    elif _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::uint32_t __result{};
                 asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_INTRINSICS() || _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp4a_u8x4_u8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp4a_s8x4_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_INTRINSICS()
  NV_IF_TARGET(
    NV_PROVIDES_SM_61,
    (return ::__dp4a(static_cast<::cuda::std::int32_t>(__lhs), static_cast<::cuda::std::int32_t>(__rhs), __acc);))
#    elif _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_INTRINSICS() || _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp4a_s8x4_s8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp4a_u8x4_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp4a_u8x4_s8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp4a_s8x4_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp4a_s8x4_u8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __dp2a_lo_u16x2_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_INTRINSICS()
  NV_IF_TARGET(NV_PROVIDES_SM_61, (return ::__dp2a_lo(__lhs, __rhs, __acc);))
#    elif _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::uint32_t __result{};
                 asm("dp2a.lo.u32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_INTRINSICS() || _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_lo_u16x2_u8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::uint32_t __dp2a_hi_u16x2_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_INTRINSICS()
  NV_IF_TARGET(NV_PROVIDES_SM_61, (return ::__dp2a_hi(__lhs, __rhs, __acc);))
#    elif _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::uint32_t __result{};
                 asm("dp2a.hi.u32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_INTRINSICS() || _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_hi_u16x2_u8x4: Unsupported architecture");
  return ::cuda::std::uint32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp2a_lo_s16x2_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_INTRINSICS()
  NV_IF_TARGET(
    NV_PROVIDES_SM_61,
    (return ::__dp2a_lo(static_cast<::cuda::std::int32_t>(__lhs), static_cast<::cuda::std::int32_t>(__rhs), __acc);))
#    elif _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp2a.lo.s32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_INTRINSICS() || _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_lo_s16x2_s8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp2a_lo_u16x2_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp2a.lo.u32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_lo_u16x2_s8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp2a_lo_s16x2_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp2a.lo.s32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_lo_s16x2_u8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp2a_hi_s16x2_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_INTRINSICS()
  NV_IF_TARGET(
    NV_PROVIDES_SM_61,
    (return ::__dp2a_hi(static_cast<::cuda::std::int32_t>(__lhs), static_cast<::cuda::std::int32_t>(__rhs), __acc);))
#    elif _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp2a.hi.s32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_INTRINSICS() || _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_hi_s16x2_s8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp2a_hi_u16x2_s8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp2a.hi.u32.s32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_hi_u16x2_s8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int32_t __dp2a_hi_s16x2_u8x4(
  [[maybe_unused]] const ::cuda::std::uint32_t __lhs,
  [[maybe_unused]] const ::cuda::std::uint32_t __rhs,
  [[maybe_unused]] const ::cuda::std::int32_t __acc) noexcept
{
#    if _CCCL_HAS_SIMD_IDOT_PTX()
  NV_IF_TARGET(NV_PROVIDES_SM_61, ({
                 ::cuda::std::int32_t __result{};
                 asm("dp2a.hi.s32.u32 %0, %1, %2, %3;" : "=r"(__result) : "r"(__lhs), "r"(__rhs), "r"(__acc));
                 return __result;
               }))
#    endif // _CCCL_HAS_SIMD_IDOT_PTX()
  _CCCL_VERIFY(false, "cuda::__simd::__dp2a_hi_s16x2_u8x4: Unsupported architecture");
  return ::cuda::std::int32_t{};
}

#  endif // _CCCL_HAS_SIMD_IDOT()

_CCCL_END_NAMESPACE_CUDA_SIMD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_SIMD_SAT() || _CCCL_HAS_SIMD_VABSDIFF() || _CCCL_HAS_SIMD_IDOT()
#endif // _CUDA___SIMD_SIMD_INTRINSICS_H
