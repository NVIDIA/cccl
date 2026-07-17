// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_arch.cuh>

#include <cuda/std/limits>

#include <string>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

inline constexpr int warp_threads                     = cub::detail::warp_threads;
inline constexpr int num_iterations                   = 100;
inline constexpr int block_dim_for_throughput_mode    = 128;
inline constexpr int grid_threads_for_throughput_mode = 1 << 28;

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

template <Mode mode>
constexpr int calc_block_dim()
{
  return (mode == Mode::Latency) ? warp_threads : block_dim_for_throughput_mode;
}

template <Mode mode>
constexpr int calc_grid_dim(int block_dim)
{
  return (mode == Mode::Latency) ? 1 : grid_threads_for_throughput_mode / block_dim;
}

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int Len>
void run_bench(nvbench::state& state)
{
  constexpr int items_per_thread = Len / warp_threads;
  const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int>;

  constexpr int block_dim = calc_block_dim<mode>();
  constexpr int grid_dim  = calc_grid_dim<mode>(block_dim);
  state.add_element_count(static_cast<size_t>(grid_dim) * (block_dim / warp_threads) * Len * num_iterations);

  state.exec([grid_dim, block_dim, kernel](nvbench::launch& launch) {
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
