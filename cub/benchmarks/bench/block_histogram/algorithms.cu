// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_histogram.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

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
  skewed,
  uniform,
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
  _CCCL_DEVICE_API _CCCL_FORCEINLINE SampleT operator()(SampleT thread_data) const
  {
    using block_histogram_t = cub::BlockHistogram<SampleT, BlockThreads, ItemsPerThread, Bins, Algorithm>;

    static_cast<void>(thread_data);

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

template <typename SampleT,
          typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm,
          sample_pattern Pattern>
void run_algorithm(nvbench::state& state)
{
  constexpr int unroll_factor = 32;
  using action_t              = benchmark_op_t<CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, Pattern>;
  const auto& kernel          = benchmark_kernel<BlockThreads, unroll_factor, action_t, SampleT>;
  const int num_sms           = state.get_device().value().get_number_of_sms();
  int max_blocks_per_sm       = 0;

  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, BlockThreads, 0));

  const int grid_size = max_blocks_per_sm * num_sms;
  state.add_element_count(
    static_cast<::cuda::std::size_t>(grid_size) * static_cast<::cuda::std::size_t>(BlockThreads)
    * static_cast<::cuda::std::size_t>(ItemsPerThread) * static_cast<::cuda::std::size_t>(unroll_factor));

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, BlockThreads>>>(action_t{});
  });
}

template <typename SampleT,
          typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm>
void dispatch_pattern(nvbench::state& state, sample_pattern pattern)
{
  switch (pattern)
  {
    case sample_pattern::single_bin:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::single_bin>(
        state);
      return;
    case sample_pattern::skewed:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::skewed>(state);
      return;
    case sample_pattern::uniform:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm, sample_pattern::uniform>(state);
      return;
  }
}

template <typename SampleT, typename CounterT, int BlockThreads, int ItemsPerThread, int Bins>
void dispatch_algorithm(nvbench::state& state, block_histogram_algorithm algorithm, sample_pattern pattern)
{
  switch (algorithm)
  {
    case block_histogram_algorithm::atomic:
      dispatch_pattern<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC>(state, pattern);
      return;
    case block_histogram_algorithm::warp_aggregated:
      dispatch_pattern<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED>(
        state, pattern);
      return;
    case block_histogram_algorithm::sort:
      dispatch_pattern<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_SORT>(state, pattern);
      return;
  }
}

template <typename SampleT, typename CounterT, int BlockThreads, int ItemsPerThread>
void dispatch_bins(nvbench::state& state, block_histogram_algorithm algorithm, sample_pattern pattern, int bins)
{
  switch (bins)
  {
    case 32:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 32>(state, algorithm, pattern);
      return;
    case 128:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 128>(state, algorithm, pattern);
      return;
    case 2048:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 2048>(state, algorithm, pattern);
      return;
  }

  throw std::runtime_error("Unsupported Bins axis value");
}

template <typename SampleT, typename CounterT, int BlockThreads>
void dispatch_items(
  nvbench::state& state, block_histogram_algorithm algorithm, sample_pattern pattern, int bins, int items_per_thread)
{
  switch (items_per_thread)
  {
    case 1:
      dispatch_bins<SampleT, CounterT, BlockThreads, 1>(state, algorithm, pattern, bins);
      return;
    case 4:
      dispatch_bins<SampleT, CounterT, BlockThreads, 4>(state, algorithm, pattern, bins);
      return;
    case 8:
      dispatch_bins<SampleT, CounterT, BlockThreads, 8>(state, algorithm, pattern, bins);
      return;
  }

  throw std::runtime_error("Unsupported ItemsPerThread axis value");
}

template <typename SampleT, typename CounterT>
void block_histogram_algorithms(nvbench::state& state, nvbench::type_list<SampleT, CounterT>)
{
  constexpr int block_threads = 256;

  const auto algorithm       = parse_algorithm(state.get_string("Algorithm"));
  const auto pattern         = parse_sample_pattern(state.get_string("SamplePattern"));
  const int bins             = static_cast<int>(state.get_int64("Bins"));
  const int items_per_thread = static_cast<int>(state.get_int64("ItemsPerThread{io}"));

  dispatch_items<SampleT, CounterT, block_threads>(state, algorithm, pattern, bins, items_per_thread);
}

using sample_types  = nvbench::type_list<::cuda::std::int32_t>;
using counter_types = nvbench::type_list<::cuda::std::int32_t>;

NVBENCH_BENCH_TYPES(block_histogram_algorithms, NVBENCH_TYPE_AXES(sample_types, counter_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}"})
  .add_string_axis("Algorithm", {"atomic", "warp_aggregated", "sort"})
  .add_int64_axis("ItemsPerThread{io}", {1, 4, 8})
  .add_int64_axis("Bins", {32, 128, 2048})
  .add_string_axis("SamplePattern", {"single_bin", "skewed", "uniform"});
