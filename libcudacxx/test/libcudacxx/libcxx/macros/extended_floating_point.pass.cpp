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
#if defined(_CCCL_HAS_NVFP16)
#  include <cuda_fp16.h>
#endif
#if defined(_CCCL_HAS_NVBF16)
#  include <cuda_bf16.h>
#endif

int main(int, char**)
{
#if _CCCL_HAS_NVFP8()
  auto x = __nv_fp8_e4m3(1.0f);
  unused(x);
#endif
#if defined(_CCCL_HAS_NVFP16)
  auto y = __half(1.0f);
  unused(y);
#endif
#if defined(_CCCL_HAS_NVBF16)
  auto z = __nv_bfloat16(1.0f);
  unused(z);
#endif
  return 0;
}
