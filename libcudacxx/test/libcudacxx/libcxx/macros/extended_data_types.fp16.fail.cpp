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

#if !_CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif

int main(int, char**)
{
#if !_CCCL_HAS_NVFP16()
  auto x2 = __half(1.0f);
  unused(x2);
#else
  static_assert(false);
#endif
  return 0;
}
