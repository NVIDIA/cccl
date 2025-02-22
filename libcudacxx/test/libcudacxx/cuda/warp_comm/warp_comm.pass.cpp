//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/warp_comm>

__device__ void warp_shuffle_semantic_test()
{
  uint32_t data = threadIdx.x;
  bool p;
  for (int i = 0; i < 32; i++)
  {
    assert(cuda::warp_shuffle(data, i) == __shfl_sync(0xFFFFFFFF, data, i));
  }
}

template <class T>
__device__ void type_test(const T& data)
{}

__device__ void type_test()
{
  type_test(cuda::std::array<double, 4>{1.0, 2.0, 3.0, 4.0});
}

__global__ void test_kernel()
{
  warp_shuffle_semantic_test();
  type_test();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
