// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_arch.cuh>

#include <cuda/std/limits>

#include <cstddef>
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

struct RunParams
{
  int block_dim;
  int grid_dim;
  int num_iterations;
};

template <Mode BenchMode>
[[nodiscard]] constexpr RunParams get_run_params()
{
  if constexpr (BenchMode == Mode::Latency)
  {
    return {warp_threads, 1, num_iterations_for_latency_mode};
  }
  else
  {
    return {block_dim_for_throughput_mode,
            grid_threads_for_throughput_mode / block_dim_for_throughput_mode,
            num_iterations_for_throughput_mode};
  }
}

[[nodiscard]] constexpr std::size_t count_items(const RunParams& run_params, int items_per_warp)
{
  return static_cast<std::size_t>(run_params.grid_dim) * (run_params.block_dim / warp_threads) * items_per_warp
       * run_params.num_iterations;
}

template <typename ActionT, Mode BenchMode, typename KeyT, typename ValueT, int Len>
void run_bench(nvbench::state& state)
{
  constexpr int items_per_thread = Len / warp_threads;
  const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int>;
  constexpr RunParams run_params = get_run_params<BenchMode>();

  state.add_element_count(count_items(run_params, Len));

  state.exec([kernel, &run_params](nvbench::launch& launch) {
    kernel<<<run_params.grid_dim, run_params.block_dim, 0, launch.get_stream()>>>(
      run_params.num_iterations, ActionT{}, Len);
  });
}

template <typename ActionT, Mode BenchMode, typename KeyT, typename ValueT, int Len, int MaxK>
void run_topk(nvbench::state& state)
{
  static_assert(MaxK >= 1);
  if constexpr (MaxK > Len)
  {
    state.skip("Skipping workload where max_k > len.");
  }
  else
  {
    constexpr int items_per_thread = Len / warp_threads;
    const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT>;
    RunParams run_params           = get_run_params<BenchMode>();
    if (BenchMode == Mode::Throughput)
    {
      // scale grid_dim because throughout mode is slow
      run_params.grid_dim /= Len / warp_threads;
    }

    state.add_element_count(count_items(run_params, Len));

    state.exec([=](nvbench::launch& launch) {
      kernel<<<run_params.grid_dim, run_params.block_dim, 0, launch.get_stream()>>>(run_params.num_iterations, ActionT{});
    });
  }
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
