// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/cmath>
#include <cuda/ptx>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/utility>

template <typename T>
__device__ __forceinline__ static T generate_random_data()
{
  constexpr auto size = static_cast<int>(cuda::ceil_div(sizeof(T), sizeof(uint32_t)));
  uint32_t data[size];
  for (int i = 0; i < size; i++)
  {
    data[i] = cuda::ptx::get_sreg_clock();
  }
  T ret;
  ::cuda::std::memcpy(&ret, data, sizeof(T));
  return ret;
}

// When benchmarking algorithms sensitive to data distribution, each thread should use a different seed to get different
// data.
template <typename T>
__device__ __forceinline__ static T generate_random_data(uint32_t& seed)
{
  constexpr auto size = static_cast<int>(cuda::ceil_div(sizeof(T), sizeof(uint32_t)));
  uint32_t data[size];
  for (int i = 0; i < size; i++)
  {
    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    seed    = 1664525 * seed + 1013904223;
    data[i] = seed;
  }
  T ret;
  cuda::std::memcpy(&ret, data, sizeof(T));
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

template <typename T, int Size>
__device__ __forceinline__ static void sink(T (&values)[Size])
{
  if (cuda::ptx::get_sreg_smid() == static_cast<uint32_t>(-1))
  {
    // use float instead of T to ensure summation order matters (due to floating-point non-associativity), making
    // `action` less likely to be optimized away
    float sum(0.0f);
    for (int i = 0; i < Size; ++i)
    {
      sum += values[i];
    }
    *reinterpret_cast<float*>(device_var) += sum;
  }
}

template <int ThreadsPerBlock, int UnrollFactor, typename ActionT, typename T>
__launch_bounds__(ThreadsPerBlock) __global__ static void benchmark_kernel(const ActionT action)
{
  auto data = generate_random_data<T>();
  cuda::static_for<UnrollFactor>([&]([[maybe_unused]] auto _) {
    data = action(data);
  });
  sink(data);
}

// This variant uses pragma directive to prevent loop unrolling, which can cause high register pressure and skew
// benchmark results.
// For keys-only benchmarks, set ValueT to void.
template <int ItemsPerThread, typename KeyT, typename ValueT, typename ActionT, typename... Args>
__global__ static void benchmark_kernel(int num_iterations, const ActionT action, Args... args)
{
  constexpr int warp_threads = 32;
  constexpr bool has_values  = !cuda::std::is_void_v<ValueT>;
  KeyT keys[ItemsPerThread];
  // when ValueT=void, declare values as char array
  [[maybe_unused]] cuda::std::conditional_t<has_values, ValueT, char> values[ItemsPerThread];
  const auto tid = threadIdx.x;
  // Shift tid by 7 to reduce the likelihood of threads within a warp getting monotonically increasing data
  uint32_t seed = cuda::ptx::get_sreg_clock() + (tid + 7) % warp_threads;

#pragma unroll 1
  for (int iter = 0; iter < num_iterations; ++iter)
  {
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      keys[i] = generate_random_data<KeyT>(seed);
      if constexpr (has_values)
      {
        values[i] = generate_random_data<ValueT>(seed);
      }
    }

    if constexpr (has_values)
    {
      action(keys, values, args...);
    }
    else
    {
      action(keys, args...);
    }

    sink(keys);
    if constexpr (has_values)
    {
      sink(values);
    }
  }
}
