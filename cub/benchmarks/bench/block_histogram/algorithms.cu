// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_histogram.cuh>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <nvbench_helper.cuh>

enum class block_histogram_algorithm
{
  atomic,
  warp_aggregated,
  warp_aggregated_cg,
  sort,
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
  if (name == "warp_aggregated_cg")
  {
    return block_histogram_algorithm::warp_aggregated_cg;
  }
  if (name == "sort")
  {
    return block_histogram_algorithm::sort;
  }

  throw std::runtime_error("Unsupported Algorithm axis value");
}

template <typename SampleT,
          typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm>
__global__ void block_histogram_kernel(const SampleT* input, CounterT* histograms)
{
  using block_histogram_t = cub::BlockHistogram<SampleT, BlockThreads, ItemsPerThread, Bins, Algorithm>;

  __shared__ typename block_histogram_t::TempStorage temp_storage;

  SampleT items[ItemsPerThread];
  const int tile_base = static_cast<int>(blockIdx.x) * BlockThreads * ItemsPerThread;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    items[item] = input[tile_base + threadIdx.x + item * BlockThreads];
  }

  CounterT* block_histogram = histograms + static_cast<std::size_t>(blockIdx.x) * Bins;
  block_histogram_t(temp_storage).Histogram(items, block_histogram);
}

template <typename SampleT,
          typename CounterT,
          int BlockThreads,
          int ItemsPerThread,
          int Bins,
          cub::BlockHistogramAlgorithm Algorithm>
void run_algorithm(nvbench::state& state, const SampleT* input, CounterT* histograms, int num_blocks)
{
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    block_histogram_kernel<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, Algorithm>
      <<<num_blocks, BlockThreads, 0, launch.get_stream().get_stream()>>>(input, histograms);
  });
}

template <typename SampleT, typename CounterT, int BlockThreads, int ItemsPerThread, int Bins>
void dispatch_algorithm(
  nvbench::state& state, block_histogram_algorithm algorithm, const SampleT* input, CounterT* histograms, int num_blocks)
{
  switch (algorithm)
  {
    case block_histogram_algorithm::atomic:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC>(
        state, input, histograms, num_blocks);
      return;
    case block_histogram_algorithm::warp_aggregated:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED>(
        state, input, histograms, num_blocks);
      return;
    case block_histogram_algorithm::warp_aggregated_cg:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED_CG>(
        state, input, histograms, num_blocks);
      return;
    case block_histogram_algorithm::sort:
      run_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_SORT>(
        state, input, histograms, num_blocks);
      return;
  }
}

template <typename SampleT, typename CounterT, int BlockThreads, int ItemsPerThread>
void dispatch_bins(
  nvbench::state& state,
  block_histogram_algorithm algorithm,
  int bins,
  const SampleT* input,
  CounterT* histograms,
  int num_blocks)
{
  switch (bins)
  {
    case 32:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 32>(
        state, algorithm, input, histograms, num_blocks);
      return;
    case 128:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 128>(
        state, algorithm, input, histograms, num_blocks);
      return;
    case 2048:
      dispatch_algorithm<SampleT, CounterT, BlockThreads, ItemsPerThread, 2048>(
        state, algorithm, input, histograms, num_blocks);
      return;
  }

  throw std::runtime_error("Unsupported Bins axis value");
}

template <typename SampleT, typename CounterT, int BlockThreads>
void dispatch_items(
  nvbench::state& state,
  int items_per_thread,
  int bins,
  block_histogram_algorithm algorithm,
  const SampleT* input,
  CounterT* histograms,
  int num_blocks)
{
  switch (items_per_thread)
  {
    case 1:
      dispatch_bins<SampleT, CounterT, BlockThreads, 1>(state, algorithm, bins, input, histograms, num_blocks);
      return;
    case 4:
      dispatch_bins<SampleT, CounterT, BlockThreads, 4>(state, algorithm, bins, input, histograms, num_blocks);
      return;
    case 8:
      dispatch_bins<SampleT, CounterT, BlockThreads, 8>(state, algorithm, bins, input, histograms, num_blocks);
      return;
  }

  throw std::runtime_error("Unsupported ItemsPerThread axis value");
}

template <typename SampleT, typename CounterT>
void block_histogram_algorithms(nvbench::state& state, nvbench::type_list<SampleT, CounterT>)
{
  constexpr int block_threads = 256;

  const auto algorithm       = parse_algorithm(state.get_string("Algorithm"));
  const auto entropy         = str_to_entropy(state.get_string("Entropy"));
  const int bins             = static_cast<int>(state.get_int64("Bins"));
  const int items_per_thread = static_cast<int>(state.get_int64("ItemsPerThread{io}"));
  const int num_blocks       = static_cast<int>(state.get_int64("Blocks"));
  const auto items_per_block = block_threads * items_per_thread;
  const auto elements        = static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(items_per_block);
  const auto histogram_items = static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(bins);

  thrust::device_vector<SampleT> input = generate(elements, entropy, SampleT{0}, static_cast<SampleT>(bins - 1));
  thrust::device_vector<CounterT> histograms(histogram_items);

  const SampleT* d_input = thrust::raw_pointer_cast(input.data());
  CounterT* d_histograms = thrust::raw_pointer_cast(histograms.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<SampleT>(elements, "InputSize");
  state.add_global_memory_writes<CounterT>(histogram_items, "HistogramSize");

  dispatch_items<SampleT, CounterT, block_threads>(
    state, items_per_thread, bins, algorithm, d_input, d_histograms, num_blocks);
}

using sample_types  = nvbench::type_list<std::int32_t>;
using counter_types = nvbench::type_list<std::int32_t>;

NVBENCH_BENCH_TYPES(block_histogram_algorithms, NVBENCH_TYPE_AXES(sample_types, counter_types))
  .set_name("base")
  .set_type_axes_names({"SampleT{ct}", "CounterT{ct}"})
  .add_string_axis("Algorithm", {"atomic", "warp_aggregated", "warp_aggregated_cg", "sort"})
  .add_int64_axis("ItemsPerThread{io}", {1, 4, 8})
  .add_int64_axis("Bins", {32, 128, 2048})
  .add_int64_axis("Blocks", {4096, 16384})
  .add_string_axis("Entropy", {"0.000", "0.201", "1.000"});
