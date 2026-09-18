// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/thread/thread_sort.cuh>

#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <cuda/ptx>
#include <cuda/std/functional>

#include <cstddef>
#include <cstdint>
#include <string>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

enum class sort_algorithm
{
  stable,
  unstable
};

enum class benchmark_mode
{
  latency,
  throughput
};

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  sort_algorithm,
  [](sort_algorithm value) {
    switch (value)
    {
      case sort_algorithm::stable:
        return "stable";
      case sort_algorithm::unstable:
        return "unstable";
    }
    return "unknown";
  },
  [](auto) {
    return std::string{};
  })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  benchmark_mode,
  [](benchmark_mode value) {
    switch (value)
    {
      case benchmark_mode::latency:
        return "latency";
      case benchmark_mode::throughput:
        return "throughput";
    }
    return "unknown";
  },
  [](auto) {
    return std::string{};
  })

inline constexpr int latency_iterations    = 100;
inline constexpr int throughput_iterations = 10;
inline constexpr int throughput_threads    = 1 << 20;
inline constexpr int threads_per_block     = 128;

template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE void mix_seed(std::uint32_t& seed, T value)
{
  seed = seed * 1664525u + static_cast<std::uint32_t>(value) + 1013904223u;
}

template <sort_algorithm Algorithm, typename KeyT, int ItemsPerThread>
_CCCL_KERNEL_ATTRIBUTES void keys_kernel(int iterations, KeyT* output)
{
  const int tid      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  std::uint32_t seed = cuda::ptx::get_sreg_clock() + static_cast<std::uint32_t>(tid);
  KeyT keys[ItemsPerThread];
  cub::NullType values[ItemsPerThread];

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration)
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys[item] = generate_random_data<KeyT>(seed);
    }

    if constexpr (Algorithm == sort_algorithm::stable)
    {
      cub::StableOddEvenSort(keys, values, cuda::std::less<>{});
    }
    else
    {
      cub::UnstablePairwiseSort(keys, values, cuda::std::less<>{});
    }

    for (int item = 0; item < ItemsPerThread; ++item)
    {
      mix_seed(seed, keys[item]);
    }
  }

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    output[tid * ItemsPerThread + item] = keys[item];
  }
}

template <sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread>
_CCCL_KERNEL_ATTRIBUTES void pairs_kernel(int iterations, KeyT* keys_output, ValueT* values_output)
{
  const int tid      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  std::uint32_t seed = cuda::ptx::get_sreg_clock() + static_cast<std::uint32_t>(tid);
  KeyT keys[ItemsPerThread];
  ValueT values[ItemsPerThread];

#pragma unroll 1
  for (int iteration = 0; iteration < iterations; ++iteration)
  {
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys[item]   = generate_random_data<KeyT>(seed);
      values[item] = generate_random_data<ValueT>(seed);
    }

    if constexpr (Algorithm == sort_algorithm::stable)
    {
      cub::StableOddEvenSort(keys, values, cuda::std::less<>{});
    }
    else
    {
      cub::UnstablePairwiseSort(keys, values, cuda::std::less<>{});
    }

    for (int item = 0; item < ItemsPerThread; ++item)
    {
      mix_seed(seed, keys[item]);
      mix_seed(seed, values[item]);
    }
  }

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const int offset      = tid * ItemsPerThread + item;
    keys_output[offset]   = keys[item];
    values_output[offset] = values[item];
  }
}

template <benchmark_mode Mode>
struct launch_config
{
  static constexpr int block_threads = Mode == benchmark_mode::latency ? 1 : threads_per_block;
  static constexpr int grid_threads  = Mode == benchmark_mode::latency ? 1 : throughput_threads;
  static constexpr int iterations    = Mode == benchmark_mode::latency ? latency_iterations : throughput_iterations;
  static constexpr int blocks        = grid_threads / block_threads;
};

template <sort_algorithm Algorithm, benchmark_mode Mode, typename KeyT, int ItemsPerThread>
void run_keys_benchmark(nvbench::state& state)
{
  using config = launch_config<Mode>;
  thrust::device_vector<KeyT> output(config::grid_threads * ItemsPerThread, thrust::no_init);
  KeyT* output_ptr = thrust::raw_pointer_cast(output.data());

  state.add_element_count(
    static_cast<std::size_t>(config::grid_threads) * static_cast<std::size_t>(ItemsPerThread) * config::iterations);
  state.exec([=](nvbench::launch& launch) {
    keys_kernel<Algorithm, KeyT, ItemsPerThread>
      <<<config::blocks, config::block_threads, 0, launch.get_stream()>>>(config::iterations, output_ptr);
  });
}

template <sort_algorithm Algorithm, benchmark_mode Mode, typename KeyT, typename ValueT, int ItemsPerThread>
void run_pairs_benchmark(nvbench::state& state)
{
  using config = launch_config<Mode>;
  thrust::device_vector<KeyT> keys_output(config::grid_threads * ItemsPerThread, thrust::no_init);
  thrust::device_vector<ValueT> values_output(config::grid_threads * ItemsPerThread, thrust::no_init);
  KeyT* keys_output_ptr     = thrust::raw_pointer_cast(keys_output.data());
  ValueT* values_output_ptr = thrust::raw_pointer_cast(values_output.data());

  state.add_element_count(
    static_cast<std::size_t>(config::grid_threads) * static_cast<std::size_t>(ItemsPerThread) * config::iterations);
  state.exec([=](nvbench::launch& launch) {
    pairs_kernel<Algorithm, KeyT, ValueT, ItemsPerThread>
      <<<config::blocks, config::block_threads, 0, launch.get_stream()>>>(
        config::iterations, keys_output_ptr, values_output_ptr);
  });
}

using modes       = nvbench::enum_type_list<benchmark_mode::latency, benchmark_mode::throughput>;
using algorithms  = nvbench::enum_type_list<sort_algorithm::stable, sort_algorithm::unstable>;
using key_types   = nvbench::type_list<std::uint16_t, std::uint32_t>;
using value_types = nvbench::type_list<std::uint32_t, std::uint64_t>;
using item_counts = nvbench::enum_type_list<4, 8, 16, 32, 64>;

template <benchmark_mode Mode, sort_algorithm Algorithm, typename KeyT, int ItemsPerThread>
void keys(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Mode>, nvbench::enum_type<Algorithm>, KeyT, nvbench::enum_type<ItemsPerThread>>)
{
  run_keys_benchmark<Algorithm, Mode, KeyT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(keys, NVBENCH_TYPE_AXES(modes, algorithms, key_types, item_counts))
  .set_type_axes_names({"mode", "algorithm", "KeyT", "items"});

template <benchmark_mode Mode, sort_algorithm Algorithm, typename KeyT, typename ValueT, int ItemsPerThread>
void pairs(
  nvbench::state& state,
  nvbench::
    type_list<nvbench::enum_type<Mode>, nvbench::enum_type<Algorithm>, KeyT, ValueT, nvbench::enum_type<ItemsPerThread>>)
{
  run_pairs_benchmark<Algorithm, Mode, KeyT, ValueT, ItemsPerThread>(state);
}

NVBENCH_BENCH_TYPES(pairs, NVBENCH_TYPE_AXES(modes, algorithms, key_types, value_types, item_counts))
  .set_type_axes_names({"mode", "algorithm", "KeyT", "ValueT", "items"});
