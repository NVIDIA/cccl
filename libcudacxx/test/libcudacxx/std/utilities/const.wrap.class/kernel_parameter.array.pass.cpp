//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: Remove once constant_wrapper is exposed.

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::__type
// UNSUPPORTED: gcc-10 || gcc-11

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
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
  for (int i = 0; i < 9; ++i)
  {
    int result;
    asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(lhs[i]), "r"(rhs[i]));
    assert(result == 9);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({
                 //  constexpr int lhs[]{1, 2, 3, 4, 5, 6, 7, 8};
                 //  constexpr int rhs[]{8, 7, 6, 5, 4, 3, 2, 1};

                 // todo(dabayer): this call crashes with error: invalid device function
                 //  test_kernel<<<1, 1>>>(cuda::std::__cw<lhs>, cuda::std::__cw<rhs>);
                 assert(cudaDeviceSynchronize() == cudaSuccess);
               }))
  return 0;
}
