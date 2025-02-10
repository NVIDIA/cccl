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

#if !_CCCL_HAS_NVFP8()
#  include <cuda_fp8.h>
#endif

int main(int, char**)
{
#if !_CCCL_HAS_NVFP8()
  auto x1 = __nv_fp8_e4m3(1.0f);
  unused(x1);
#else
  static_assert(false);
#endif
  return 0;
}
