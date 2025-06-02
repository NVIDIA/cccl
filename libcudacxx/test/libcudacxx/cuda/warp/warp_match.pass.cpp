//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: pre-sm-70

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/warp>

#include "test_macros.h"

template <typename T>
__device__ void test_types(T valueA = T{}, T valueB = T{1})
{
  for (int i = 1; i < 32; ++i)
  {
    auto mask = cuda::device::lane_mask{(1u << i) - 1};
    assert(cuda::device::warp_match_all(valueA, mask));
    auto value = threadIdx.x == 0 ? valueA : valueB;
    assert(!cuda::device::warp_match_all(value, mask));
  }
}

__global__ void test_kernel()
{
  test_types<uint8_t>();
  test_types<uint16_t>();
  test_types<uint32_t>();
  test_types<uint64_t>();
#if _CCCL_HAS_INT128()
  test_types<__uint128_t>();
#endif
  test_types(char3{0, 0, 0}, char3{1, 1, 1});
  using array_t = cuda::std::array<char, 6>;
  test_types(array_t{0, 0, 0, 0, 0, 0}, array_t{1, 1, 1, 1, 1, 1});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
