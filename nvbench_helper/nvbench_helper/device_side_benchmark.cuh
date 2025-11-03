// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/utility>

template <typename T>
__device__ __forceinline__ static T generate_random_data()
{
  constexpr auto size = cuda::ceil_div(sizeof(T), sizeof(uint32_t));
  uint32_t data[size];
  for (int i = 0; i < size; i++)
  {
    data[i] = cuda::ptx::get_sreg_clock();
  }
  T ret;
  ::cuda::std::memcpy(&ret, data, sizeof(T));
  return ret;
}

__device__ static int device_var[16];

template <typename T>
__device__ __forceinline__ static void sink(T value)
{
  if (cuda::ptx::get_sreg_smid() == static_cast<uint32_t>(-1))
  {
    *reinterpret_cast<T*>(device_var) = value;
  }
}

template <int BlockThreads, int UnrollFactor, typename ActionT, typename T>
__launch_bounds__(BlockThreads) __global__ static void benchmark_kernel(_CCCL_GRID_CONSTANT const ActionT action)
{
  auto data = generate_random_data<T>();
  cuda::static_for<UnrollFactor>([&]([[maybe_unused]] auto _) {
    data = action(data);
  });
  sink(data);
}
