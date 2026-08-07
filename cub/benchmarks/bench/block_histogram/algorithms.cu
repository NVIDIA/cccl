// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_histogram.cuh>

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

enum class block_histogram_algorithm
{
  atomic,
  warp_aggregated,
  sort,
};

enum class sample_pattern
{
  single_bin,
  lane_pairs,
  skewed,
  uniform,
};

enum class benchmark_mode
{
  occupancy,
  fixed,
  // Used by the separate latency benchmark, not by the BenchmarkMode axis.
  latency,
  full_bounds,
};

block_histogram_algorithm parse_algorithm(const std::string& name)
{
  if (name == "atomic")
  {
    return block_histogram_algorithm::atomic;
  }
  if (name == "warp_aggregated")
  {
    return block_histogram_algorithm::warp_aggregated;
  }
  if (name == "sort")
  {
    return block_histogram_algorithm::sort;
  }

  throw std::runtime_error("Unsupported Algorithm axis value");
}

sample_pattern parse_sample_pattern(const std::string& name)
{
  if (name == "single_bin")
  {
    return sample_pattern::single_bin;
  }
  if (name == "lane_pairs")
  {
    return sample_pattern::lane_pairs;
  }
  if (name == "skewed")
  {
    return sample_pattern::skewed;
  }
  if (name == "uniform")
  {
    return sample_pattern::uniform;
  }

  throw std::runtime_error("Unsupported SamplePattern axis value");
}

benchmark_mode parse_benchmark_mode(const std::string& name)
{
  if (name == "occupancy")
  {
    return benchmark_mode::occupancy;
  }
  if (name == "fixed")
  {
    return benchmark_mode::fixed;
  }
  if (name == "full_bounds")
  {
    return benchmark_mode::full_bounds;
  }

  throw std::runtime_error("Unsupported BenchmarkMode axis value");
}

template <int BlockThreads, int Bins, sample_pattern Pattern, typename SampleT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE SampleT make_sample(int item)
{
  constexpr int warp_threads = cub::detail::warp_threads;
  const int lane             = threadIdx.x % warp_threads;
  const int striped_offset   = threadIdx.x + item * BlockThreads;

  if constexpr (Pattern == sample_pattern::single_bin)
  {
    return SampleT{0};
  }
  else if constexpr (Pattern == sample_pattern::lane_pairs)
  {
    return static_cast<SampleT>((lane / 2) % Bins);
  }
  else if constexpr (Pattern == sample_pattern::skewed)
  {
    return static_cast<SampleT>(lane < 24 ? 0 : striped_offset % Bins);
  }
  else
  {
    return static_cast<SampleT>(striped_offset % Bins);
  }
}

template <typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm,
          sample_pattern Pattern>
struct benchmark_op_t
{
  template <typename SampleT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE SampleT operator()(SampleT) const
  {
    using block_histogram_t = cub::BlockHistogram<SampleT, BlockThreads, ItemsPerThread, Bins, Algorithm>;

    __shared__ typename block_histogram_t::TempStorage temp_storage;
    __shared__ CounterT histogram[Bins];

    SampleT items[ItemsPerThread];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      items[item] = make_sample<BlockThreads, Bins, Pattern, SampleT>(item);
    }

    block_histogram_t(temp_storage).Histogram(items, histogram);
    __syncthreads();

    const auto sink_bin = (threadIdx.x + blockIdx.x * blockDim.x) % Bins;
    return static_cast<SampleT>(histogram[sink_bin]);
  }
};

template <std::size_t Numerator, std::size_t Denominator>
inline constexpr std::size_t static_ceil_div = (Numerator + Denominator - 1) / Denominator;

template <int BlockThreads, int SmBlocks, int UnrollFactor, typename ActionT, typename T>
__launch_bounds__(BlockThreads, SmBlocks) __global__
  static void kernel_full_bounds(_CCCL_GRID_CONSTANT const ActionT action)
{
  auto data = generate_random_data<T>();
  cuda::static_for<UnrollFactor>([&]([[maybe_unused]] auto _) {
    data = action(data);
  });
  sink(data);
}

template <typename SampleT,
          typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm,
          sample_pattern Pattern,
          benchmark_mode Mode>
void run_algorithm(nvbench::state& state)
{
  constexpr int unroll_factor = 32;
  using action_t              = benchmark_op_t<CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, Pattern>;
  const auto& kernel          = benchmark_kernel<BlockThreads, unroll_factor, action_t, SampleT>;

  if constexpr (Mode == benchmark_mode::latency)
  {
    constexpr int grid_size = 1;
    state.add_element_count(static_cast<std::size_t>(grid_size) * BlockThreads * ItemsPerThread * unroll_factor);
    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_size, BlockThreads, 0, launch.get_stream()>>>(action_t{});
    });
    return;
  }
  else
  {
    constexpr std::size_t fixed_work_items = std::size_t{1} << 28;
    constexpr std::size_t items_per_block  = BlockThreads * ItemsPerThread * unroll_factor;
    constexpr int fixed_grid_size          = static_cast<int>(static_ceil_div<fixed_work_items, items_per_block>);

    if constexpr (Mode == benchmark_mode::full_bounds)
    {
      // Use a launch-bound target that is valid for the oldest SMs in the benchmark matrix.
      constexpr int resident_threads_per_sm   = 1024;
      constexpr int full_bounds_max_sm_blocks = resident_threads_per_sm / BlockThreads;
      const auto& full_bounds_kernel =
        kernel_full_bounds<BlockThreads, full_bounds_max_sm_blocks, unroll_factor, action_t, SampleT>;

      state.add_element_count(
        static_cast<std::size_t>(fixed_grid_size) * BlockThreads * ItemsPerThread * unroll_factor);
      state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
        full_bounds_kernel<<<fixed_grid_size, BlockThreads, 0, launch.get_stream()>>>(action_t{});
      });
      return;
    }

    int grid_size = fixed_grid_size;

    if constexpr (Mode == benchmark_mode::occupancy)
    {
      const int num_sms     = state.get_device().value().get_number_of_sms();
      int max_blocks_per_sm = 0;

      NVBENCH_CUDA_CALL_NOEXCEPT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, BlockThreads, 0));

      grid_size = max_blocks_per_sm * num_sms;
    }

    state.add_element_count(static_cast<std::size_t>(grid_size) * BlockThreads * ItemsPerThread * unroll_factor);

    state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
      kernel<<<grid_size, BlockThreads, 0, launch.get_stream()>>>(action_t{});
    });
  }
}

template <typename SampleT,
          typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm,
          benchmark_mode Mode>
void dispatch_pattern(nvbench::state& state, sample_pattern pattern)
{
  switch (pattern)
  {
    case sample_pattern::single_bin:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::single_bin, Mode>(
        state);
      return;
    case sample_pattern::lane_pairs:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::lane_pairs, Mode>(
        state);
      return;
    case sample_pattern::skewed:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::skewed, Mode>(
        state);
      return;
    case sample_pattern::uniform:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::uniform, Mode>(
        state);
      return;
  }
}

template <typename SampleT, typename CounterT, int BlockThreads, int ItemsPerThread, int Bins, benchmark_mode Mode>
void dispatch_algorithm(nvbench::state& state, block_histogram_algorithm algorithm, sample_pattern pattern)
{
  switch (algorithm)
  {
    case block_histogram_algorithm::atomic:
      dispatch_pattern<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC, Mode>(
        state, pattern);
      return;
    case block_histogram_algorithm::warp_aggregated:
      dispatch_pattern<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED, Mode>(
        state, pattern);
      return;
    case block_histogram_algorithm::sort:
      dispatch_pattern<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_SORT, Mode>(
        state, pattern);
      return;
  }
}

template <typename SampleT, typename CounterT, int BlockThreads, int ItemsPerThread, benchmark_mode Mode>
void dispatch_bins(nvbench::state& state, block_histogram_algorithm algorithm, sample_pattern pattern, int bins)
{
  switch (bins)
  {
    case 32:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 32, Mode>(state, algorithm, pattern);
      return;
    case 64:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 64, Mode>(state, algorithm, pattern);
      return;
    case 128:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 128, Mode>(state, algorithm, pattern);
      return;
    case 512:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 512, Mode>(state, algorithm, pattern);
      return;
    case 2048:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 2048, Mode>(state, algorithm, pattern);
      return;
  }

  throw std::runtime_error("Unsupported Bins axis value");
}

template <typename SampleT, typename CounterT, int BlockThreads, benchmark_mode Mode>
void dispatch_items(
  nvbench::state& state, block_histogram_algorithm algorithm, sample_pattern pattern, int bins, int items_per_thread)
{
  switch (items_per_thread)
  {
    case 1:
      dispatch_bins<SampleT, CounterT, BlockThreads, 1, Mode>(state, algorithm, pattern, bins);
      return;
    case 4:
      dispatch_bins<SampleT, CounterT, BlockThreads, 4, Mode>(state, algorithm, pattern, bins);
      return;
    case 8:
      dispatch_bins<SampleT, CounterT, BlockThreads, 8, Mode>(state, algorithm, pattern, bins);
      return;
  }

  throw std::runtime_error("Unsupported ItemsPerThread axis value");
}

template <typename SampleT, typename CounterT, benchmark_mode Mode>
void dispatch_block_threads(
  nvbench::state& state,
  block_histogram_algorithm algorithm,
  sample_pattern pattern,
  int bins,
  int items_per_thread,
  int block_threads)
{
  switch (block_threads)
  {
    case 128:
      dispatch_items<SampleT, CounterT, 128, Mode>(state, algorithm, pattern, bins, items_per_thread);
      return;
    case 256:
      dispatch_items<SampleT, CounterT, 256, Mode>(state, algorithm, pattern, bins, items_per_thread);
      return;
    case 512:
      dispatch_items<SampleT, CounterT, 512, Mode>(state, algorithm, pattern, bins, items_per_thread);
      return;
  }

  throw std::runtime_error("Unsupported BlockThreads axis value");
}

template <typename SampleT, typename CounterT>
void dispatch_mode(
  nvbench::state& state,
  block_histogram_algorithm algorithm,
  sample_pattern pattern,
  benchmark_mode mode,
  int bins,
  int items_per_thread,
  int block_threads)
{
  switch (mode)
  {
    case benchmark_mode::occupancy:
      dispatch_block_threads<SampleT, CounterT, benchmark_mode::occupancy>(
        state, algorithm, pattern, bins, items_per_thread, block_threads);
      return;
    case benchmark_mode::fixed:
      dispatch_block_threads<SampleT, CounterT, benchmark_mode::fixed>(
        state, algorithm, pattern, bins, items_per_thread, block_threads);
      return;
    case benchmark_mode::full_bounds:
      dispatch_block_threads<SampleT, CounterT, benchmark_mode::full_bounds>(
        state, algorithm, pattern, bins, items_per_thread, block_threads);
      return;
    case benchmark_mode::latency:
      // The latency benchmark bypasses dispatch_mode and dispatches this mode directly with one warp.
      break;
  }

  throw std::runtime_error("Unsupported BenchmarkMode axis value");
}

template <typename SampleT, typename CounterT>
void block_histogram_algorithms(nvbench::state& state, nvbench::type_list<SampleT, CounterT>)
{
  const auto algorithm       = parse_algorithm(state.get_string("Algorithm"));
  const auto pattern         = parse_sample_pattern(state.get_string("SamplePattern"));
  const auto mode            = parse_benchmark_mode(state.get_string("BenchmarkMode"));
  const int bins             = static_cast<int>(state.get_int64("Bins"));
  const int items_per_thread = static_cast<int>(state.get_int64("ItemsPerThread{io}"));
  const int block_threads    = static_cast<int>(state.get_int64("BlockThreads"));

  dispatch_mode<SampleT, CounterT>(state, algorithm, pattern, mode, bins, items_per_thread, block_threads);
}

template <typename SampleT, typename CounterT>
void block_histogram_algorithms_latency(nvbench::state& state, nvbench::type_list<SampleT, CounterT>)
{
  const auto algorithm       = parse_algorithm(state.get_string("Algorithm"));
  const auto pattern         = parse_sample_pattern(state.get_string("SamplePattern"));
  const int bins             = static_cast<int>(state.get_int64("Bins"));
  const int items_per_thread = static_cast<int>(state.get_int64("ItemsPerThread{io}"));

  dispatch_items<SampleT, CounterT, cub::detail::warp_threads, benchmark_mode::latency>(
    state, algorithm, pattern, bins, items_per_thread);
}

using sample_types  = nvbench::type_list<int>;
using counter_types = nvbench::type_list<int>;

NVBENCH_BENCH_TYPES(block_histogram_algorithms, NVBENCH_TYPE_AXES(sample_types, counter_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}"})
  .add_string_axis("Algorithm", {"atomic", "warp_aggregated", "sort"})
  .add_string_axis("BenchmarkMode", {"occupancy", "fixed", "full_bounds"})
  .add_int64_axis("BlockThreads", {128, 256, 512})
  .add_int64_axis("ItemsPerThread{io}", {1, 4, 8})
  .add_int64_axis("Bins", {32, 64, 128, 512, 2048})
  .add_string_axis("SamplePattern", {"single_bin", "lane_pairs", "skewed", "uniform"});

NVBENCH_BENCH_TYPES(block_histogram_algorithms_latency, NVBENCH_TYPE_AXES(sample_types, counter_types))
  .set_name("latency")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}"})
  .add_string_axis("Algorithm", {"atomic", "warp_aggregated", "sort"})
  .add_int64_axis("ItemsPerThread{io}", {1, 4, 8})
  .add_int64_axis("Bins", {32, 64, 128, 512, 2048})
  .add_string_axis("SamplePattern", {"single_bin", "lane_pairs", "skewed", "uniform"});
