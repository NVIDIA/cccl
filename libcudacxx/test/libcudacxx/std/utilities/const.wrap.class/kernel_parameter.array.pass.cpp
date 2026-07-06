//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// nvcc generates different symbol on host and device leading to kernel launch failure. Seems to be working with gcc as
// the host compiler.
// UNSUPPORTED: nvhpc || (clang && nvcc)

// nvrtc is unsupported.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// Test that constant_wrapper can be safely passed as a parameter from host to device.

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

template <class Lhs, class Rhs>
__global__ void test_kernel(Lhs lhs, Rhs rhs)
{
  for (int i = 0; i < 8; ++i)
  {
    int result;
    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(lhs[i]), "r"(rhs[i]));
    assert(result == 9);
  }
}

TEST_GLOBAL_VARIABLE int lhs[8]{1, 2, 3, 4, 5, 6, 7, 8};
TEST_GLOBAL_VARIABLE int rhs[8]{8, 7, 6, 5, 4, 3, 2, 1};

void test_host()
{
  test_kernel<<<1, 1>>>(cuda::std::__cw<lhs>, cuda::std::__cw<rhs>);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_host();))
  return 0;
}
