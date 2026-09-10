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
  // launch grid_threads_for_throughput_mode threads and report Elem/s.
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
  int grid_dim;
  int block_dim;
  int num_iterations;
};

template <Mode mode>
[[nodiscard]] constexpr RunParams calc_run_params()
{
  if constexpr (mode == Mode::Latency)
  {
    return {1, warp_threads, num_iterations_for_latency_mode};
  }
  else
  {
    return {grid_threads_for_throughput_mode / block_dim_for_throughput_mode,
            block_dim_for_throughput_mode,
            num_iterations_for_throughput_mode};
  }
}

template <Mode mode>
[[nodiscard]] constexpr RunParams calc_topk_run_params(int items_per_warp)
{
  RunParams result = calc_run_params<mode>();
  if constexpr (mode == Mode::Throughput)
  {
    // Scale the grid inversely with input length because TopK throughput runs are long and stable.
    result.grid_dim /= items_per_warp / warp_threads;
  }
  return result;
}

[[nodiscard]] constexpr std::size_t count_items(const RunParams& run_params, int items_per_warp)
{
  return static_cast<std::size_t>(run_params.grid_dim) * (run_params.block_dim / warp_threads) * items_per_warp
       * run_params.num_iterations;
}

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int Len>
void run_bench(nvbench::state& state)
{
  constexpr int items_per_thread = Len / warp_threads;
  const auto kernel              = benchmark_kernel<items_per_thread, KeyT, ValueT, ActionT, int>;
  constexpr RunParams run_params = calc_run_params<mode>();

  state.add_element_count(count_items(run_params, Len));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    kernel<<<run_params.grid_dim, run_params.block_dim, 0, launch.get_stream()>>>(
      run_params.num_iterations, ActionT{}, Len);
  });
}

template <typename ActionT, Mode mode, typename KeyT, typename ValueT, int Len, int MaxK>
void run_topk_bench(nvbench::state& state)
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
    constexpr RunParams run_params = calc_topk_run_params<mode>(Len);

    state.add_element_count(count_items(run_params, Len));

    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<run_params.grid_dim, run_params.block_dim, 0, launch.get_stream()>>>(
        run_params.num_iterations, ActionT{});
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
