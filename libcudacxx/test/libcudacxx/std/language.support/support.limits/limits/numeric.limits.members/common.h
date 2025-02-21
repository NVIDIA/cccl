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

// Disable all the extended floating point operations and conversions
#define __CUDA_NO_FP4_CONVERSIONS__          1
#define __CUDA_NO_FP4_CONVERSION_OPERATORS__ 1
#define __CUDA_NO_FP6_CONVERSIONS__          1
#define __CUDA_NO_FP6_CONVERSION_OPERATORS__ 1
#define __CUDA_NO_FP8_CONVERSIONS__          1
#define __CUDA_NO_FP8_CONVERSION_OPERATORS__ 1
#define __CUDA_NO_HALF_CONVERSIONS__         1
#define __CUDA_NO_HALF_OPERATORS__           1
#define __CUDA_NO_BFLOAT16_CONVERSIONS__     1
#define __CUDA_NO_BFLOAT16_OPERATORS__       1

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/limits>

template <class T>
__host__ __device__ bool float_eq(T x, T y)
{
  return x == y;
}

#if _CCCL_HAS_NVFP4()
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
#endif // _CCCL_HAS_NVFP4

#if _CCCL_HAS_NVFP6()
__host__ __device__ inline __nv_fp6_e2m3 make_fp6_e2m3(double x)
{
  __nv_fp6_e2m3 res;
  res.__x = __nv_cvt_double_to_fp6(x, __NV_E2M3, cudaRoundNearest);
  return res;
}

__host__ __device__ inline __nv_fp6_e3m2 make_fp6_e3m2(double x)
{
  __nv_fp6_e3m2 res;
  res.__x = __nv_cvt_double_to_fp6(x, __NV_E3M2, cudaRoundNearest);
  return res;
}

__host__ __device__ inline bool float_eq(__nv_fp6_e2m3 x, __nv_fp6_e2m3 y)
{
  return x.__x == y.__x;
}

__host__ __device__ inline bool float_eq(__nv_fp6_e3m2 x, __nv_fp6_e3m2 y)
{
  return x.__x == y.__x;
}
#endif // _CCCL_HAS_NVFP6

#if _CCCL_HAS_NVFP8()
__host__ __device__ inline __nv_fp8_e4m3 make_fp8_e4m3(double x, __nv_saturation_t sat = __NV_NOSAT)
{
  __nv_fp8_e4m3 res;
  res.__x = __nv_cvt_double_to_fp8(x, sat, __NV_E4M3);
  return res;
}

__host__ __device__ inline __nv_fp8_e5m2 make_fp8_e5m2(double x, __nv_saturation_t sat = __NV_NOSAT)
{
  __nv_fp8_e5m2 res;
  res.__x = __nv_cvt_double_to_fp8(x, sat, __NV_E5M2);
  return res;
}

#  if _CCCL_CUDACC_AT_LEAST(12, 8)
__host__ __device__ inline __nv_fp8_e8m0 make_fp8_e8m0(double x, __nv_saturation_t sat = __NV_NOSAT)
{
  __nv_fp8_e8m0 res;
  res.__x = __nv_cvt_double_to_e8m0(x, sat, cudaRoundZero);
  return res;
}
#  endif // _CCCL_CUDACC_AT_LEAST(12, 8)

__host__ __device__ inline bool float_eq(__nv_fp8_e4m3 x, __nv_fp8_e4m3 y)
{
  return x.__x == y.__x;
}

__host__ __device__ inline bool float_eq(__nv_fp8_e5m2 x, __nv_fp8_e5m2 y)
{
  return x.__x == y.__x;
}

#  if _CCCL_CUDACC_AT_LEAST(12, 8)
__host__ __device__ inline bool float_eq(__nv_fp8_e8m0 x, __nv_fp8_e8m0 y)
{
  return x.__x == y.__x;
}
#  endif // _CCCL_CUDACC_AT_LEAST(12, 8)
#endif // _CCCL_HAS_NVFP8

#if _CCCL_HAS_NVFP16()
__host__ __device__ inline bool float_eq(__half x, __half y)
{
#  if _CCCL_CUDACC_AT_LEAST(12, 2)
  return __heq(x, y);
#  else
  return __half2float(x) == __half2float(y);
#  endif
}
#endif // _CCCL_HAS_NVFP16

#if _CCCL_HAS_NVBF16()
__host__ __device__ inline bool float_eq(__nv_bfloat16 x, __nv_bfloat16 y)
{
#  if _CCCL_CUDACC_AT_LEAST(12, 2)
  return __heq(x, y);
#  else
  return __bfloat162float(x) == __bfloat162float(y);
#  endif
}
#endif // _CCCL_HAS_NVBF16

#endif // NUMERIC_LIMITS_MEMBERS_COMMON_H
