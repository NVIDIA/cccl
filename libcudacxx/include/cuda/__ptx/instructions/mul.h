// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_MUL_H_
#define _CUDA_PTX_MUL_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/ptx_dot_variants.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

_CCCL_DEVICE static inline _CUDA_VSTD::int16_t
mul(_CUDA_VPTX::mul_mode_lo_t, _CUDA_VSTD::int16_t __x, _CUDA_VSTD::int16_t __y)
{
  _CUDA_VSTD::int16_t __result;
  asm("mul.lo.s16 %0, %1, %2;" : "=h"(__result) : "h"(__x), "h"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint16_t
mul(_CUDA_VPTX::mul_mode_lo_t, _CUDA_VSTD::uint16_t __x, _CUDA_VSTD::uint16_t __y)
{
  _CUDA_VSTD::uint16_t __result;
  asm("mul.lo.u16 %0, %1, %2;" : "=h"(__result) : "h"(__x), "h"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int32_t
mul(_CUDA_VPTX::mul_mode_lo_t, _CUDA_VSTD::int32_t __x, _CUDA_VSTD::int32_t __y)
{
  _CUDA_VSTD::int32_t __result;
  asm("mul.lo.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t
mul(_CUDA_VPTX::mul_mode_lo_t, _CUDA_VSTD::uint32_t __x, _CUDA_VSTD::uint32_t __y)
{
  _CUDA_VSTD::uint32_t __result;
  asm("mul.lo.u32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int64_t
mul(_CUDA_VPTX::mul_mode_lo_t, _CUDA_VSTD::int64_t __x, _CUDA_VSTD::int64_t __y)
{
  _CUDA_VSTD::int64_t __result;
  asm("mul.lo.s64 %0, %1, %2;" : "=l"(__result) : "l"(__x), "l"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t
mul(_CUDA_VPTX::mul_mode_lo_t, _CUDA_VSTD::uint64_t __x, _CUDA_VSTD::uint64_t __y)
{
  _CUDA_VSTD::uint64_t __result;
  asm("mul.lo.u64 %0, %1, %2;" : "=l"(__result) : "l"(__x), "l"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int16_t
mul(_CUDA_VPTX::mul_mode_hi_t, _CUDA_VSTD::int16_t __x, _CUDA_VSTD::int16_t __y)
{
  _CUDA_VSTD::int16_t __result;
  asm("mul.hi.s16 %0, %1, %2;" : "=h"(__result) : "h"(__x), "h"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint16_t
mul(_CUDA_VPTX::mul_mode_hi_t, _CUDA_VSTD::uint16_t __x, _CUDA_VSTD::uint16_t __y)
{
  _CUDA_VSTD::uint16_t __result;
  asm("mul.hi.u16 %0, %1, %2;" : "=h"(__result) : "h"(__x), "h"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int32_t
mul(_CUDA_VPTX::mul_mode_hi_t, _CUDA_VSTD::int32_t __x, _CUDA_VSTD::int32_t __y)
{
  _CUDA_VSTD::int32_t __result;
  asm("mul.hi.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t
mul(_CUDA_VPTX::mul_mode_hi_t, _CUDA_VSTD::uint32_t __x, _CUDA_VSTD::uint32_t __y)
{
  _CUDA_VSTD::uint32_t __result;
  asm("mul.hi.u32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int64_t
mul(_CUDA_VPTX::mul_mode_hi_t, _CUDA_VSTD::int64_t __x, _CUDA_VSTD::int64_t __y)
{
  _CUDA_VSTD::int64_t __result;
  asm("mul.hi.s64 %0, %1, %2;" : "=l"(__result) : "l"(__x), "l"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t
mul(_CUDA_VPTX::mul_mode_hi_t, _CUDA_VSTD::uint64_t __x, _CUDA_VSTD::uint64_t __y)
{
  _CUDA_VSTD::uint64_t __result;
  asm("mul.hi.u64 %0, %1, %2;" : "=l"(__result) : "l"(__x), "l"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int32_t
mul(_CUDA_VPTX::mul_mode_wide_t, _CUDA_VSTD::int16_t __x, _CUDA_VSTD::int16_t __y)
{
  _CUDA_VSTD::int32_t __result;
  asm("mul.wide.s16 %0, %1, %2;" : "=r"(__result) : "h"(__x), "h"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t
mul(_CUDA_VPTX::mul_mode_wide_t, _CUDA_VSTD::uint16_t __x, _CUDA_VSTD::uint16_t __y)
{
  _CUDA_VSTD::uint32_t __result;
  asm("mul.wide.u16 %0, %1, %2;" : "=r"(__result) : "h"(__x), "h"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::int64_t
mul(_CUDA_VPTX::mul_mode_wide_t, _CUDA_VSTD::int32_t __x, _CUDA_VSTD::int32_t __y)
{
  _CUDA_VSTD::int64_t __result;
  asm("mul.wide.s32 %0, %1, %2;" : "=l"(__result) : "r"(__x), "r"(__y));
  return __result;
}

_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t
mul(_CUDA_VPTX::mul_mode_wide_t, _CUDA_VSTD::uint32_t __x, _CUDA_VSTD::uint32_t __y)
{
  _CUDA_VSTD::uint64_t __result;
  asm("mul.wide.u32 %0, %1, %2;" : "=l"(__result) : "r"(__x), "r"(__y));
  return __result;
}

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_MUL_H_
