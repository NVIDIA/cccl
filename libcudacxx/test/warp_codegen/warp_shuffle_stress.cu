//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// implement a simple warp bitonic sort to check that the register usage of warp shuffle operations matches native
// shuffle instructions

#include <cuda/std/cstdint>
#include <cuda/warp> // IWYU pragma: keep

template <bool UseWrapper, typename T>
__device__ __forceinline__ T shuffle_xor(T value, int lane_mask)
{
  if constexpr (UseWrapper)
  {
    return cuda::device::warp_shuffle_xor(value, lane_mask).data;
  }
  else
  {
    return __shfl_xor_sync(0xFFFFFFFFu, value, lane_mask);
  }
}

struct key_value
{
  cuda::std::uint16_t key;
  cuda::std::uint64_t value;
};

template <bool UseWrapper>
__device__ __forceinline__ void stress_body(const key_value* input, key_value* output)
{
  constexpr int items_per_thread = 4;
  constexpr int lane_masks[]     = {1, 2, 1, 4, 2, 1, 8, 4, 2, 1, 16, 8, 4, 2, 1};
  key_value items[items_per_thread];

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item)
  {
    items[item] = input[items_per_thread * threadIdx.x + item];
  }

#pragma unroll
  for (int lane_mask : lane_masks)
  {
#pragma unroll
    for (auto& item : items)
    {
      const auto other_key   = shuffle_xor<UseWrapper>(item.key, lane_mask);
      const auto other_value = shuffle_xor<UseWrapper>(item.value, lane_mask);
      if ((threadIdx.x & lane_mask) != 0)
      {
        item = {other_key, other_value};
      }
    }
  }

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item)
  {
    output[items_per_thread * threadIdx.x + item] = items[item];
  }
}

extern "C" __global__ void wrapper_stress(const key_value* input, key_value* output)
{
  stress_body<true>(input, output);
}

extern "C" __global__ void native_stress(const key_value* input, key_value* output)
{
  stress_body<false>(input, output);
}
