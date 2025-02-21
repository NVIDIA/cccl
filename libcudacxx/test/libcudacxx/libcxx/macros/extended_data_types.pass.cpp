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

#if _CCCL_HAS_NVFP8()
#  include <cuda_fp8.h>
#endif
#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif
#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>
#endif

int main(int, char**)
{
#if _CCCL_HAS_INT128()
  auto x1 = __int128(123456789123) + __int128(123456789123);
  auto y1 = __uint128_t(123456789123) + __uint128_t(123456789123);
  unused(x1);
  unused(y1);
#endif
#if _CCCL_HAS_NVFP8()
  auto x2 = __nv_fp8_e4m3(1.0f);
  unused(x2);
#endif
#if _CCCL_HAS_NVFP16()
  auto x3 = __half(1.0f);
  unused(x3);
#endif
#if _CCCL_HAS_NVBF16()
  auto x4 = __nv_bfloat16(1.0f);
  unused(x4);
#endif
#if _CCCL_HAS_FLOAT128()
  __float128 x5 = __float128(3.14) + __float128(3.14);
  unused(x5);
#endif
  return 0;
}
