//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/__cccl/extended_data_types.h>

#include "test_macros.h"

#if _CCCL_HAS_NVFP4()
#  include <cuda_fp4.h>
#endif
#if _CCCL_HAS_NVFP6_E2M3()
#  include <cuda_fp6.h>
#endif
#if _CCCL_HAS_NVFP8()
#  include <cuda_fp8.h>
#endif
#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif
#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif

template <class T>
__host__ __device__ void test_nv_fp()
{
  auto v = T{1.0f};
  unused(v);
}

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  auto a = __int128(123456789123) + __int128(123456789123);
  auto b = __uint128_t(123456789123) + __uint128_t(123456789123);
  unused(a, b);
#endif

#if _CCCL_HAS_NVFP4_E2M1()
  test_nv_fp<__nv_fp4_e2m1>();
#endif
#if _CCCL_HAS_NVFP6_E3M2()
  test_nv_fp<__nv_fp6_e3m2>();
#endif
#if _CCCL_HAS_NVFP6_E2M3()
  test_nv_fp<__nv_fp6_e2m3>();
#endif
#if _CCCL_HAS_NVFP8_E4M3()
  test_nv_fp<__nv_fp8_e4m3>();
#endif
#if _CCCL_HAS_NVFP8_E5M2()
  test_nv_fp<__nv_fp8_e5m2>();
#endif
#if _CCCL_HAS_NVFP8_E8M0()
  test_nv_fp<__nv_fp8_e8m0>();
#endif
#if _CCCL_HAS_NVFP16()
  test_nv_fp<__half>();
#endif
#if _CCCL_HAS_NVBF16()
  test_nv_fp<__nv_bfloat16>();
#endif

#if _CCCL_HAS_FLOAT128()
  __float128 x5 = __float128(3.14) + __float128(3.14);
  unused(x5);
#endif

  return 0;
}
