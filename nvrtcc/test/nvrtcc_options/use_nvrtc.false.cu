//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "../common/check_predefined_macros.h"

#if !defined(__CUDACC_RTC__)
#  include <cassert>
#endif // !__CUDACC_RTC__

__managed__ int proof_var;

__global__ void proof_kernel()
{
#if defined(__CUDACC_RTC__)
  proof_var = 1;
#else // ^^^ __CUDACC_RTC__ ^^^ / vvv !__CUDACC_RTC__ vvv
  proof_var = -1;
#endif // ^^^ !__CUDACC_RTC__ ^^^
}

#if !defined(__CUDACC_RTC__)
int main()
{
  proof_var = 0;

  proof_kernel<<<1, 1>>>();
  assert(cudaDeviceSynchronize() == cudaSuccess);

  assert(proof_var == -1);
}
#endif // ^^^ !__CUDACC_RTC__ ^^^
