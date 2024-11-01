// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11
// UNSUPPORTED: nvrtc
// UNSUPPORTED: msvc && c++14
// UNSUPPORTED: msvc && c++17

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#include "test_macros.h"

// We are experiencing data corruption on clang when passing a mdspan mapping around where on of the subtypes is empty
struct empty
{};
struct mapping
{
  using __member_pair_t = _CUDA_VSTD::__detail::__compressed_pair<empty, int>;
  _CCCL_NO_UNIQUE_ADDRESS __member_pair_t __members;
};

__global__ void kernel(mapping arg1, mapping arg2)
{
  assert(arg1.__members.__second() == arg2.__members.__second());
}

void test()
{
  mapping strided{{empty{}, 1}};
  kernel<<<1, 1>>>(strided, strided);
  cudaDeviceSynchronize();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
