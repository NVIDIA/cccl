//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef NUMERIC_LIMITS_MEMBERS_COMMON_H
#define NUMERIC_LIMITS_MEMBERS_COMMON_H

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/limits>

template <class T>
__host__ __device__ bool float_eq(T x, T y)
{
  return x == y;
}

#if _CCCL_HAS_NVFP4_E2M1()
__host__ __device__ inline __nv_fp4_e2m1 make_fp4_e2m1(double x)
{
  __nv_fp4_e2m1 res;
  res.__x = __nv_cvt_double_to_fp4(x, __NV_E2M1, cudaRoundNearest);
  return res;
}
__host__ __device__ inline bool float_eq(__nv_fp4_e2m1 x, __nv_fp4_e2m1 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP4_E2M1()

#if _CCCL_HAS_NVFP6_E2M3()
__host__ __device__ inline __nv_fp6_e2m3 make_fp6_e2m3(double x)
{
  __nv_fp6_e2m3 res;
  res.__x = __nv_cvt_double_to_fp6(x, __NV_E2M3, cudaRoundNearest);
  return res;
}
__host__ __device__ inline bool float_eq(__nv_fp6_e2m3 x, __nv_fp6_e2m3 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
__host__ __device__ inline __nv_fp6_e3m2 make_fp6_e3m2(double x)
{
  __nv_fp6_e3m2 res;
  res.__x = __nv_cvt_double_to_fp6(x, __NV_E3M2, cudaRoundNearest);
  return res;
}
__host__ __device__ inline bool float_eq(__nv_fp6_e3m2 x, __nv_fp6_e3m2 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP8_E4M3()
__host__ __device__ inline __nv_fp8_e4m3 make_fp8_e4m3(double x, __nv_saturation_t sat = __NV_NOSAT)
{
  __nv_fp8_e4m3 res;
  res.__x = __nv_cvt_double_to_fp8(x, sat, __NV_E4M3);
  return res;
}
__host__ __device__ inline bool float_eq(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
__host__ __device__ inline __nv_fp8_e5m2 make_fp8_e5m2(double x, __nv_saturation_t sat = __NV_NOSAT)
{
  __nv_fp8_e5m2 res;
  res.__x = __nv_cvt_double_to_fp8(x, sat, __NV_E5M2);
  return res;
}
__host__ __device__ inline bool float_eq(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
__host__ __device__ inline __nv_fp8_e8m0 make_fp8_e8m0(double x, __nv_saturation_t sat = __NV_NOSAT)
{
  __nv_fp8_e8m0 res;
  res.__x = __nv_cvt_double_to_e8m0(x, sat, cudaRoundZero);
  return res;
}
__host__ __device__ inline bool float_eq(__nv_fp8_e8m0 x, __nv_fp8_e8m0 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVF8_E8M0()

#if _CCCL_HAS_NVFP16()
__host__ __device__ inline bool float_eq(__half x, __half y)
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return __heq(x, y);
#  else
  return __half2float(x) == __half2float(y);
#  endif
}
#endif // _CCCL_HAS_NVFP16

#if _CCCL_HAS_NVBF16()
__host__ __device__ inline bool float_eq(__nv_bfloat16 x, __nv_bfloat16 y)
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return __heq(x, y);
#  else
  return __bfloat162float(x) == __bfloat162float(y);
#  endif
}
#endif // _CCCL_HAS_NVBF16

#endif // NUMERIC_LIMITS_MEMBERS_COMMON_H
