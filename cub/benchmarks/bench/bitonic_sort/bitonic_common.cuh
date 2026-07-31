// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_arch.cuh>

#include <cuda/std/limits>

#include <string>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

inline constexpr int warp_threads                       = cub::detail::warp_threads;
inline constexpr int num_iterations_for_latency_mode    = 100;
inline constexpr int num_iterations_for_throughput_mode = 10;
inline constexpr int block_dim_for_throughput_mode      = 128;
inline constexpr int grid_threads_for_throughput_mode   = 1 << 27;

enum class Mode
{
  // launch a single warp
  Latency,
  // launch grid_threads_for_throughput_mode threads. Measure Elem/s.
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

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int Len>
void run_bench(nvbench::state& state)
{
  constexpr int items_per_thread = Len / warp_threads;
  const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int>;

  int block_dim;
  int grid_dim;
  int num_iterations;
  if (mode == Mode::Latency)
  {
    block_dim      = warp_threads;
    grid_dim       = 1;
    num_iterations = num_iterations_for_latency_mode;
  }
  else
  {
    block_dim      = block_dim_for_throughput_mode;
    grid_dim       = grid_threads_for_throughput_mode / block_dim;
    num_iterations = num_iterations_for_throughput_mode;
  }

  state.add_element_count(static_cast<size_t>(grid_dim) * (block_dim / warp_threads) * Len * num_iterations);

  state.exec([grid_dim, block_dim, kernel, num_iterations](nvbench::launch& launch) {
    kernel<<<grid_dim, block_dim, 0, launch.get_stream()>>>(num_iterations, ActionT{}, Len);
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
  static constexpr T oob_default =
    cuda::std::numeric_limits<T>::has_infinity
      ? cuda::std::numeric_limits<T>::infinity()
      : cuda::std::numeric_limits<T>::max();
};
