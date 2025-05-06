// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_histogram.cuh>
#include <cub/grid/grid_queue.cuh>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Histogram kernel entry points
 *****************************************************************************/

/**
 * Histogram initialization kernel entry point
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param num_output_bins_wrapper
 *   Number of output histogram bins per channel
 *
 * @param d_output_histograms_wrapper
 *   Histogram counter data having logical dimensions
 *   `CounterT[NUM_ACTIVE_CHANNELS][num_bins.array[CHANNEL]]`
 *
 * @param tile_queue
 *   Drain queue descriptor for dynamically mapping tile data onto thread blocks
 */
namespace detail::histogram
{

template <int NUM_ACTIVE_CHANNELS, typename CounterT, typename OffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceHistogramInitKernel(
  ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper,
  ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms_wrapper,
  GridQueue<int> tile_queue)
{
  if ((threadIdx.x == 0) && (blockIdx.x == 0))
  {
    tile_queue.ResetDrain();
  }

  int output_bin = (blockIdx.x * blockDim.x) + threadIdx.x;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int CHANNEL = 0; CHANNEL < NUM_ACTIVE_CHANNELS; ++CHANNEL)
  {
    if (output_bin < num_output_bins_wrapper[CHANNEL])
    {
      d_output_histograms_wrapper[CHANNEL][output_bin] = 0;
    }
  }
}

/**
 * Histogram privatized sweep kernel entry point (multi-block).
 * Computes privatized histograms, one per thread block.
 *
 *
 * @tparam AgentHistogramPolicyT
 *   Parameterized AgentHistogramPolicy tuning policy type
 *
 * @tparam PRIVATIZED_SMEM_BINS
 *   Maximum number of histogram bins per channel (e.g., up to 256)
 *
 * @tparam NUM_CHANNELS
 *   Number of channels interleaved in the input data (may be greater than the number of channels
 *   being actively histogrammed)
 *
 * @tparam NUM_ACTIVE_CHANNELS
 *   Number of channels actively being histogrammed
 *
 * @tparam SampleIteratorT
 *   The input iterator type. @iterator.
 *
 * @tparam CounterT
 *   Integer type for counting sample occurrences per histogram bin
 *
 * @tparam PrivatizedDecodeOpT
 *   The transform operator type for determining privatized counter indices from samples,
 *   one for each channel
 *
 * @tparam OutputDecodeOpT
 *   The transform operator type for determining output bin-ids from privatized counter indices,
 *   one for each channel
 *
 * @tparam OffsetT
 *   integer type for global offsets
 *
 * @param d_samples
 *   Input data to reduce
 *
 * @param num_output_bins_wrapper
 *   The number bins per final output histogram
 *
 * @param num_privatized_bins_wrapper
 *   The number bins per privatized histogram
 *
 * @param d_output_histograms_wrapper
 *   Reference to final output histograms
 *
 * @param d_privatized_histograms_wrapper
 *   Reference to privatized histograms
 *
 * @param output_decode_op_wrapper
 *   The transform operator for determining output bin-ids from privatized counter indices,
 *   one for each channel
 *
 * @param privatized_decode_op_wrapper
 *   The transform operator for determining privatized counter indices from samples,
 *   one for each channel
 *
 * @param num_row_pixels
 *   The number of multi-channel pixels per row in the region of interest
 *
 * @param num_rows
 *   The number of rows in the region of interest
 *
 * @param row_stride_samples
 *   The number of samples between starts of consecutive rows in the region of interest
 *
 * @param tiles_per_row
 *   Number of image tiles per row
 *
 * @param tile_queue
 *   Drain queue descriptor for dynamically mapping tile data onto thread blocks
 */
template <typename ChainedPolicyT,
          int PRIVATIZED_SMEM_BINS,
          int NUM_CHANNELS,
          int NUM_ACTIVE_CHANNELS,
          typename SampleIteratorT,
          typename CounterT,
          typename PrivatizedDecodeOpT,
          typename OutputDecodeOpT,
          typename OffsetT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::AgentHistogramPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceHistogramSweepKernel(
    SampleIteratorT d_samples,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_output_bins_wrapper,
    ::cuda::std::array<int, NUM_ACTIVE_CHANNELS> num_privatized_bins_wrapper,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_output_histograms_wrapper,
    ::cuda::std::array<CounterT*, NUM_ACTIVE_CHANNELS> d_privatized_histograms_wrapper,
    ::cuda::std::array<OutputDecodeOpT, NUM_ACTIVE_CHANNELS> output_decode_op_wrapper,
    ::cuda::std::array<PrivatizedDecodeOpT, NUM_ACTIVE_CHANNELS> privatized_decode_op_wrapper,
    OffsetT num_row_pixels,
    OffsetT num_rows,
    OffsetT row_stride_samples,
    int tiles_per_row,
    GridQueue<int> tile_queue)
{
  // Thread block type for compositing input tiles
  using AgentHistogramPolicyT = typename ChainedPolicyT::ActivePolicy::AgentHistogramPolicyT;
  using AgentHistogramT =
    AgentHistogram<AgentHistogramPolicyT,
                   PRIVATIZED_SMEM_BINS,
                   NUM_CHANNELS,
                   NUM_ACTIVE_CHANNELS,
                   SampleIteratorT,
                   CounterT,
                   PrivatizedDecodeOpT,
                   OutputDecodeOpT,
                   OffsetT>;

  // Shared memory for AgentHistogram
  __shared__ typename AgentHistogramT::TempStorage temp_storage;

  AgentHistogramT agent(
    temp_storage,
    d_samples,
    num_output_bins_wrapper.data(),
    num_privatized_bins_wrapper.data(),
    d_output_histograms_wrapper.data(),
    d_privatized_histograms_wrapper.data(),
    output_decode_op_wrapper.data(),
    privatized_decode_op_wrapper.data());

  // Initialize counters
  agent.InitBinCounters();

  // Consume input tiles
  agent.ConsumeTiles(num_row_pixels, num_rows, row_stride_samples, tiles_per_row, tile_queue);

  // Store output to global (if necessary)
  agent.StoreOutput();
}
} // namespace detail::histogram
CUB_NAMESPACE_END
