//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvcc + nvrtc fails to create stubs for kernels kernels that take constant_wrapper as an argument.
// UNSUPPORTED: !clang || nvcc

// REQUIRES: !c++17

// constant_wrapper

// Test that constant_wrapper can be safely passed as a parameter from host to device.

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

template <class Lhs, class Rhs>
__global__ void test_kernel(Lhs lhs, Rhs rhs)
{
  int result;
  asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(lhs.value), "r"(rhs.value));
  assert(result == 9);
}

void test_host()
{
  test_kernel<<<1, 1>>>(cuda::std::__cw<1>, cuda::std::__cw<8>);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_host();))
  return 0;
}
