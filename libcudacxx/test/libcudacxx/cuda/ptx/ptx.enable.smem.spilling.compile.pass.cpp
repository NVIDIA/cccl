//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/ptx>

#include <cuda/ptx>

__global__ void test_enable_smem_spilling()
{
#if __cccl_ptx_isa >= 900
  cuda::ptx::enable_smem_spilling();
#endif // __cccl_ptx_isa >= 900
}

int main(int, char**)
{
  return 0;
}
