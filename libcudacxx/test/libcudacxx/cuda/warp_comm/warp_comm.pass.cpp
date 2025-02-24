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
#include <cuda/std/type_traits>
#include <cuda/warp_comm>

#include "test_macros.h"

template <int Value>
inline constexpr auto width_v = cuda::std::integral_constant<int, Value>{};

template <int Value>
__device__ void test_semantic()
{
  uint32_t data           = threadIdx.x;
  constexpr uint32_t mask = 0xFFFFFFFF;
  for (int i = 0; i < 64; i++)
  {
    assert(cuda::warp_shuffle_idx(data, i, mask, width_v<Value>) == __shfl_sync(mask, data, i, Value));
  }
  for (int i = 1; i < Value; i++)
  {
    assert(cuda::warp_shuffle_down(data, i, mask, width_v<Value>) == __shfl_down_sync(mask, data, i, Value));
    assert(cuda::warp_shuffle_up(data, i, mask, width_v<Value>) == __shfl_up_sync(mask, data, i, Value));
    assert(cuda::warp_shuffle_xor(data, i, mask, width_v<Value>) == __shfl_xor_sync(mask, data, i, Value));
  }
  unused(data);
  unused(mask);
  if (Value == 16 && threadIdx.x < 16)
  {
    constexpr uint32_t mask2 = 0xFFFF;
    int i                    = 4;
    assert(cuda::warp_shuffle_idx(data, i, mask2, width_v<Value>) == __shfl_sync(mask2, data, i, Value));
    assert(cuda::warp_shuffle_down(data, i, mask2, width_v<Value>) == __shfl_down_sync(mask2, data, i, Value));
    assert(cuda::warp_shuffle_xor(data, i, mask2, width_v<Value>) == __shfl_xor_sync(mask2, data, i, Value));
    assert(cuda::warp_shuffle_up(data, i, mask2, width_v<Value>) == __shfl_up_sync(mask2, data, i, Value));
  }
}

template <class T>
__device__ void test_large_type(const T& data)
{
  auto data1 = threadIdx.x == 0 ? data : T{};
  assert(cuda::warp_shuffle_idx(data1, 0).data == data);
  auto data2 = threadIdx.x >= 2 ? data : T{};
  assert(cuda::warp_shuffle_down(data2, 2).data == data);
  auto data3 = threadIdx.x < 30 ? data : T{};
  assert(cuda::warp_shuffle_up(data3, 2).data == data);
  auto data4 = threadIdx.x % 2 == 0 ? data : T{};
  assert(cuda::warp_shuffle_xor(data4, 1).data == ((threadIdx.x % 2 == 0) ? T{} : data));
  unused(data1);
  unused(data2);
  unused(data3);
  unused(data4);
}

__device__ void test_overloadings()
{
  uint32_t data           = threadIdx.x;
  constexpr uint32_t mask = 0xFFFFFFFF;
  for (int i = 0; i < 64; i++)
  {
    assert(cuda::warp_shuffle_idx(data, i, mask) == __shfl_sync(mask, data, i));
  }
  for (int i = 1; i < 32; i++)
  {
    assert(cuda::warp_shuffle_down(data, i, mask) == __shfl_down_sync(mask, data, i));
    assert(cuda::warp_shuffle_up(data, i, mask) == __shfl_up_sync(mask, data, i));
    assert(cuda::warp_shuffle_xor(data, i, mask) == __shfl_xor_sync(mask, data, i));
  }
}

__global__ void test_kernel()
{
  test_semantic<1>();
  test_semantic<2>();
  test_semantic<4>();
  test_semantic<8>();
  test_semantic<16>();
  test_semantic<32>();
  test_overloadings();
  test_large_type(cuda::std::array<double, 4>{1.0, 2.0, 3.0, 4.0});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_kernel<<<1, 32>>>();))
  return 0;
}
