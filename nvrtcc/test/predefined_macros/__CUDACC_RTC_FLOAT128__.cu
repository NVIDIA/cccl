//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#if defined(__CUDACC_RTC__) && __CUDA_ARCH__ >= 1000
#  define EXPECT_CUDACC_RTC_FLOAT128
#endif // __CUDACC_RTC__ && __CUDA_ARCH__ >= 1000
#include "../common/check_predefined_macros.h"

#if defined(EXPECT_CUDACC_RTC_FLOAT128)
__global__ void kernel(__float128* value)
{
  *value = __float128{1.0};
}
#endif // EXPECT_CUDACC_RTC_FLOAT128
