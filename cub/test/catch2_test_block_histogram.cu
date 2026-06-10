// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Test of BlockHistogram utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <cub/block/block_histogram.cuh>

#include <limits>
#include <string>

#include <c2h/catch2_test_helper.h>

template <int Bins,
          int BlockThreads,
          int ItemsPerThread,
          cub::BlockHistogramAlgorithm Algorithm,
          typename T,
          typename HistoCounter>
__global__ void block_histogram_kernel(T* d_samples, HistoCounter* d_histogram)
{
  // Parameterize BlockHistogram type for our thread block
  using block_histogram_t = cub::BlockHistogram<T, BlockThreads, ItemsPerThread, Bins, Algorithm>;

  // Allocate temp storage in shared memory
  __shared__ typename block_histogram_t::TempStorage temp_storage;

  // Per-thread tile data
  T data[ItemsPerThread];
  cub::LoadDirectStriped<BlockThreads>(threadIdx.x, d_samples, data);

  // Test histo (writing directly to histogram buffer in global)
  block_histogram_t(temp_storage).Histogram(data, d_histogram);
}

template <int ItemsPerThread, int ThreadsInBlock, int Bins, cub::BlockHistogramAlgorithm Algorithm, typename SampleT>
void block_histogram(c2h::device_vector<SampleT>& d_samples, c2h::device_vector<int>& d_histogram)
{
  block_histogram_kernel<Bins, ThreadsInBlock, ItemsPerThread, Algorithm>
    <<<1, ThreadsInBlock>>>(thrust::raw_pointer_cast(d_samples.data()), thrust::raw_pointer_cast(d_histogram.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int Bins, int BlockThreads, int ItemsPerThread, typename T, typename HistoCounter>
__global__ void block_histogram_composite_kernel(T* d_samples, HistoCounter* d_histogram)
{
  using block_histogram_t =
    cub::BlockHistogram<T, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED>;

  __shared__ typename block_histogram_t::TempStorage temp_storage;

  T data[ItemsPerThread];
  int block_offset = blockIdx.x * BlockThreads * ItemsPerThread;
  cub::LoadDirectStriped<BlockThreads>(threadIdx.x, d_samples + block_offset, data);

  block_histogram_t(temp_storage).Composite(data, d_histogram);
}

template <int ItemsPerThread, int ThreadsInBlock, int Bins, typename SampleT>
void block_histogram_composite(
  c2h::device_vector<SampleT>& d_samples, c2h::device_vector<int>& d_histogram, int num_blocks)
{
  block_histogram_composite_kernel<Bins, ThreadsInBlock, ItemsPerThread><<<num_blocks, ThreadsInBlock>>>(
    thrust::raw_pointer_cast(d_samples.data()), thrust::raw_pointer_cast(d_histogram.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int Bins, int BlockThreads, int ItemsPerThread, typename T, typename HistoCounter>
__global__ void block_histogram_composite_shared_kernel(T* d_samples, HistoCounter* d_histogram)
{
  using block_histogram_t =
    cub::BlockHistogram<T, BlockThreads, ItemsPerThread, Bins, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED>;

  __shared__ typename block_histogram_t::TempStorage temp_storage;
  __shared__ HistoCounter s_histogram[Bins];

  for (int bin = threadIdx.x; bin < Bins; bin += BlockThreads)
  {
    s_histogram[bin] = 0;
  }

  __syncthreads();

  T data[ItemsPerThread];
  cub::LoadDirectStriped<BlockThreads>(threadIdx.x, d_samples, data);

  block_histogram_t(temp_storage).Composite(data, s_histogram);

  __syncthreads();

  for (int bin = threadIdx.x; bin < Bins; bin += BlockThreads)
  {
    d_histogram[bin] = s_histogram[bin];
  }
}

template <int ItemsPerThread, int ThreadsInBlock, int Bins, typename SampleT>
void block_histogram_composite_shared(c2h::device_vector<SampleT>& d_samples, c2h::device_vector<int>& d_histogram)
{
  block_histogram_composite_shared_kernel<Bins, ThreadsInBlock, ItemsPerThread>
    <<<1, ThreadsInBlock>>>(thrust::raw_pointer_cast(d_samples.data()), thrust::raw_pointer_cast(d_histogram.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// %PARAM% TEST_BINS bins 32:256:1024

using types            = c2h::type_list<std::uint8_t, std::uint16_t>;
using threads_in_block = c2h::enum_type_list<int, 32, 33, 96, 128>;
using items_per_thread = c2h::enum_type_list<int, 1, 5>;
using bins             = c2h::enum_type_list<int, TEST_BINS>;
using algorithms =
  c2h::enum_type_list<cub::BlockHistogramAlgorithm,
                      cub::BLOCK_HISTO_SORT,
                      cub::BLOCK_HISTO_ATOMIC,
                      cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED>;
using warp_aggregated_algorithm =
  c2h::enum_type_list<cub::BlockHistogramAlgorithm, cub::BLOCK_HISTO_ATOMIC_WARP_AGGREGATED>;
using composite_items_per_thread        = c2h::enum_type_list<int, 3>;
using global_composite_threads_in_block = c2h::enum_type_list<int, 64>;
using shared_composite_threads_in_block = c2h::enum_type_list<int, 33>;

template <class TestType>
struct params_t
{
  using sample_t = typename c2h::get<0, TestType>;

  static constexpr int items_per_thread                   = c2h::get<1, TestType>::value;
  static constexpr int threads_in_block                   = c2h::get<2, TestType>::value;
  static constexpr int bins                               = c2h::get<3, TestType>::value;
  static constexpr int num_samples                        = threads_in_block * items_per_thread;
  static constexpr cub::BlockHistogramAlgorithm algorithm = c2h::get<4, TestType>::value;
};

C2H_TEST("Block histogram can be computed with uniform input",
         "[histogram][block]",
         types,
         items_per_thread,
         threads_in_block,
         bins,
         algorithms)
{
  using params   = params_t<TestType>;
  using sample_t = typename params::sample_t;

  const sample_t uniform_value = static_cast<sample_t>(GENERATE_COPY(take(10, random(0, params::bins - 1))));

  c2h::host_vector<sample_t> h_samples(params::num_samples, uniform_value);
  c2h::host_vector<int> h_reference(params::bins);
  h_reference[static_cast<std::size_t>(uniform_value)] = params::num_samples;

  // Allocate problem device arrays
  c2h::device_vector<sample_t> d_samples = h_samples;
  c2h::device_vector<int> d_histogram(params::bins);

  // Run kernel
  block_histogram<params::items_per_thread, params::threads_in_block, params::bins, params::algorithm>(
    d_samples, d_histogram);

  REQUIRE(h_reference == d_histogram);
}

template <typename SampleT>
c2h::host_vector<int> compute_host_reference(int bins, const c2h::host_vector<SampleT>& h_samples)
{
  c2h::host_vector<int> h_reference(bins);
  for (const SampleT& sample : h_samples)
  {
    h_reference[sample]++;
  }

  return h_reference;
}

C2H_TEST("Block histogram can be computed with modulo input",
         "[histogram][block]",
         types,
         items_per_thread,
         threads_in_block,
         bins,
         algorithms)
{
  using params   = params_t<TestType>;
  using sample_t = typename params::sample_t;

  // Allocate problem device arrays
  c2h::device_vector<int> d_histogram(params::bins);
  c2h::device_vector<sample_t> d_samples(params::num_samples);

  c2h::gen(c2h::modulo_t{params::bins}, d_samples);

  c2h::host_vector<sample_t> h_samples = d_samples;
  auto h_reference                     = compute_host_reference(params::bins, h_samples);

  // Run kernel
  block_histogram<params::items_per_thread, params::threads_in_block, params::bins, params::algorithm>(
    d_samples, d_histogram);

  REQUIRE(h_reference == d_histogram);
}

C2H_TEST("Block histogram can be computed with random input",
         "[histogram][block]",
         types,
         items_per_thread,
         threads_in_block,
         bins,
         algorithms)
{
  using params   = params_t<TestType>;
  using sample_t = typename params::sample_t;

  // Allocate problem device arrays
  c2h::device_vector<int> d_histogram(params::bins);
  c2h::device_vector<sample_t> d_samples(params::num_samples);

  const sample_t min_bin = static_cast<sample_t>(0);
  const sample_t max_bin = static_cast<sample_t>(
    std::min(static_cast<std::int32_t>(cuda::std::numeric_limits<sample_t>::max()),
             static_cast<std::int32_t>(params::bins - 1)));

  c2h::gen(C2H_SEED(10), d_samples, min_bin, max_bin);

  c2h::host_vector<sample_t> h_samples = d_samples;
  auto h_reference                     = compute_host_reference(params::bins, h_samples);

  // Run kernel
  block_histogram<params::items_per_thread, params::threads_in_block, params::bins, params::algorithm>(
    d_samples, d_histogram);

  REQUIRE(h_reference == d_histogram);
}

C2H_TEST("Block histogram warp-aggregated composite updates a global histogram from multiple blocks",
         "[histogram][block]",
         types,
         composite_items_per_thread,
         global_composite_threads_in_block,
         bins,
         warp_aggregated_algorithm)
{
  using params   = params_t<TestType>;
  using sample_t = typename params::sample_t;

  static constexpr int num_blocks  = 5;
  static constexpr int num_samples = num_blocks * params::num_samples;

  c2h::host_vector<sample_t> h_samples(num_samples);
  for (int i = 0; i < num_samples; ++i)
  {
    h_samples[i] = static_cast<sample_t>(i % params::bins);
  }

  auto h_reference = compute_host_reference(params::bins, h_samples);

  c2h::device_vector<sample_t> d_samples = h_samples;
  c2h::device_vector<int> d_histogram(params::bins, 0);

  block_histogram_composite<params::items_per_thread, params::threads_in_block, params::bins>(
    d_samples, d_histogram, num_blocks);

  REQUIRE(h_reference == d_histogram);
}

C2H_TEST("Block histogram warp-aggregated composite updates a shared histogram from a partial warp block",
         "[histogram][block]",
         types,
         composite_items_per_thread,
         shared_composite_threads_in_block,
         bins,
         warp_aggregated_algorithm)
{
  using params   = params_t<TestType>;
  using sample_t = typename params::sample_t;

  static constexpr int num_samples = params::num_samples;

  c2h::host_vector<sample_t> h_samples(num_samples);
  for (int i = 0; i < num_samples; ++i)
  {
    h_samples[i] = static_cast<sample_t>(i % params::bins);
  }

  auto h_reference = compute_host_reference(params::bins, h_samples);

  c2h::device_vector<sample_t> d_samples = h_samples;
  c2h::device_vector<int> d_histogram(params::bins, 0);

  block_histogram_composite_shared<params::items_per_thread, params::threads_in_block, params::bins>(
    d_samples, d_histogram);

  REQUIRE(h_reference == d_histogram);
}
