// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/limits>

#include <string>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

constexpr int WARP_THREADS                  = 32;
constexpr int NUM_ITERATIONS                = 100;
constexpr int BLOCK_DIM_FOR_THROUGHPUT_MODE = 128;

enum class Mode
{
  // launch single warp
  Latency,
  // launch one full wave of thread blocks. Measure Elem/s.
  Throughput
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  Mode,
  // Callable to generate input strings:
  [](Mode value) {
    switch (value)
    {
      case Mode::Latency:
        return "latency";
      case Mode::Throughput:
        return "throughput";
      default:
        return "Unknown";
    }
  },
  // Callable to generate descriptions:
  [](auto) {
    return std::string{};
  })

template <Mode mode>
constexpr int calc_block_dim()
{
  if constexpr (mode == Mode::Latency)
  {
    return WARP_THREADS;
  }
  else
  {
    return BLOCK_DIM_FOR_THROUGHPUT_MODE;
  }
}

template <Mode mode, typename Kernel>
int calc_grid_dim(int num_SMs, int block_dim, Kernel kernel)
{
  if constexpr (mode == Mode::Latency)
  {
    return 1;
  }
  else
  {
    int max_blocks_per_SM = 0;
    NVBENCH_CUDA_CALL_NOEXCEPT(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_SM, kernel, block_dim, 0));
    return max_blocks_per_SM * num_SMs;
  }
}

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int LEN>
void run_bench(nvbench::state& state)
{
  constexpr int items_per_thread = LEN / WARP_THREADS;
  const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int>;

  const int num_SMs       = state.get_device().value().get_number_of_sms();
  constexpr int block_dim = calc_block_dim<mode>();
  const int grid_dim      = calc_grid_dim<mode>(num_SMs, block_dim, kernel);
  state.add_element_count(grid_dim * (block_dim / WARP_THREADS) * LEN * NUM_ITERATIONS);

  state.exec([grid_dim, block_dim, kernel](nvbench::launch& launch) {
    kernel<<<grid_dim, block_dim, 0, launch.get_stream()>>>(NUM_ITERATIONS, ActionT{}, LEN);
  });
}

struct CustomLess
{
  template <typename T>
  __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }

  template <typename T>
  static constexpr T oob = cuda::std::numeric_limits<T>::max();
};
