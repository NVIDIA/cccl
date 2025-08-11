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

#include <cub/agent/agent_segmented_radix_sort.cuh>
#include <cub/agent/agent_sub_warp_merge_sort.cuh>
#include <cub/detail/device_double_buffer.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_device.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

CUB_NAMESPACE_BEGIN
namespace detail::segmented_sort
{

// Type used to index within segments within a single invocation
using local_segment_index_t = ::cuda::std::uint32_t;
// Type used for total number of segments and to index within segments globally
using global_segment_offset_t = ::cuda::std::int64_t;

/**
 * @brief Fallback kernel, in case there's not enough segments to
 *        take advantage of partitioning.
 *
 * In this case, the sorting method is still selected based on the segment size.
 * If a single warp can sort the segment, the algorithm will use the sub-warp
 * merge sort. Otherwise, the algorithm will use the in-shared-memory version of
 * block radix sort. If data don't fit into shared memory, the algorithm will
 * use in-global-memory radix sort.
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in,out] d_keys_double_buffer
 *   Double keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in,out] d_values_double_buffer
 *   Double values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   i-th data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   i-th data segment in `d_keys_*` and `d_values_*`.
 *   If `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the i-th segment is
 *   considered empty.
 */
template <SortOrder Order,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::LargeSegmentPolicy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortFallbackKernel(
    const KeyT* d_keys_in_orig,
    KeyT* d_keys_out_orig,
    device_double_buffer<KeyT> d_keys_double_buffer,
    const ValueT* d_values_in_orig,
    ValueT* d_values_out_orig,
    device_double_buffer<ValueT> d_values_double_buffer,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets)
{
  using ActivePolicyT       = typename ChainedPolicyT::ActivePolicy;
  using LargeSegmentPolicyT = typename ActivePolicyT::LargeSegmentPolicy;
  using MediumPolicyT       = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT::MediumPolicyT;

  const auto segment_id = static_cast<local_segment_index_t>(blockIdx.x);
  OffsetT segment_begin = d_begin_offsets[segment_id];
  OffsetT segment_end   = d_end_offsets[segment_id];
  OffsetT num_items     = segment_end - segment_begin;

  if (num_items <= 0)
  {
    return;
  }

  using AgentSegmentedRadixSortT =
    radix_sort::AgentSegmentedRadixSort<Order == SortOrder::Descending, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;

  using WarpReduceT = cub::WarpReduce<KeyT>;

  using AgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, MediumPolicyT, KeyT, ValueT, OffsetT>;

  __shared__ union
  {
    typename AgentSegmentedRadixSortT::TempStorage block_sort;
    typename WarpReduceT::TempStorage warp_reduce;
    typename AgentWarpMergeSortT::TempStorage medium_warp_sort;
  } temp_storage;

  constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  AgentSegmentedRadixSortT agent(num_items, temp_storage.block_sort);

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  constexpr int cacheable_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;

  d_keys_in_orig += segment_begin;
  d_keys_out_orig += segment_begin;

  if (!keys_only)
  {
    d_values_in_orig += segment_begin;
    d_values_out_orig += segment_begin;
  }

  if (num_items <= MediumPolicyT::ITEMS_PER_TILE)
  {
    // Sort by a single warp
    if (threadIdx.x < MediumPolicyT::WARP_THREADS)
    {
      AgentWarpMergeSortT(temp_storage.medium_warp_sort)
        .ProcessSegment(num_items, d_keys_in_orig, d_keys_out_orig, d_values_in_orig, d_values_out_orig);
    }
  }
  else if (num_items < cacheable_tile_size)
  {
    // Sort by a CTA if data fits into shared memory
    agent.ProcessSinglePass(begin_bit, end_bit, d_keys_in_orig, d_values_in_orig, d_keys_out_orig, d_values_out_orig);
  }
  else
  {
    // Sort by a CTA with multiple reads from global memory
    int current_bit = begin_bit;
    int pass_bits   = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

    d_keys_double_buffer = device_double_buffer<KeyT>(
      d_keys_double_buffer.current() + segment_begin, d_keys_double_buffer.alternate() + segment_begin);

    if (!keys_only)
    {
      d_values_double_buffer = device_double_buffer<ValueT>(
        d_values_double_buffer.current() + segment_begin, d_values_double_buffer.alternate() + segment_begin);
    }

    agent.ProcessIterative(
      current_bit,
      pass_bits,
      d_keys_in_orig,
      d_values_in_orig,
      d_keys_double_buffer.current(),
      d_values_double_buffer.current());
    current_bit += pass_bits;

    _CCCL_PRAGMA_NOUNROLL()
    while (current_bit < end_bit)
    {
      pass_bits = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

      __syncthreads();
      agent.ProcessIterative(
        current_bit,
        pass_bits,
        d_keys_double_buffer.current(),
        d_values_double_buffer.current(),
        d_keys_double_buffer.alternate(),
        d_values_double_buffer.alternate());

      d_keys_double_buffer.swap();
      d_values_double_buffer.swap();
      current_bit += pass_bits;
    }
  }
}

/**
 * @brief Single kernel for moderate size (less than a few thousand items)
 *        segments.
 *
 * This kernel allocates a sub-warp per segment. Therefore, this kernel assigns
 * a single thread block to multiple segments. Segments fall into two
 * categories. An architectural warp usually sorts segments in the medium-size
 * category, while a few threads sort segments in the small-size category. Since
 * segments are partitioned, we know the last thread block index assigned to
 * sort medium-size segments. A particular thread block can check this number to
 * find out which category it was assigned to sort. In both cases, the
 * merge sort is used.
 *
 * @param[in] small_segments
 *   Number of segments that can be sorted by a warp part
 *
 * @param[in] medium_segments
 *   Number of segments that can be sorted by a warp
 *
 * @param[in] medium_blocks
 *   Number of CTAs assigned to process medium segments
 *
 * @param[in] d_small_segments_indices
 *   Small segments mapping of length @p small_segments, such that
 *   `d_small_segments_indices[i]` is the input segment index
 *
 * @param[in] d_medium_segments_indices
 *   Medium segments mapping of length @p medium_segments, such that
 *   `d_medium_segments_indices[i]` is the input segment index
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If
 *   `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the <em>i</em><sup>th</sup> is
 *   considered empty.
 */
template <SortOrder Order,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::SmallAndMediumSegmentedSortPolicyT::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortKernelSmall(
    local_segment_index_t small_segments,
    local_segment_index_t medium_segments,
    local_segment_index_t medium_blocks,
    const local_segment_index_t* d_small_segments_indices,
    const local_segment_index_t* d_medium_segments_indices,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets)
{
  using local_segment_index_t = local_segment_index_t;

  const local_segment_index_t tid = threadIdx.x;
  const local_segment_index_t bid = blockIdx.x;

  using ActivePolicyT         = typename ChainedPolicyT::ActivePolicy;
  using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;
  using MediumPolicyT         = typename SmallAndMediumPolicyT::MediumPolicyT;
  using SmallPolicyT          = typename SmallAndMediumPolicyT::SmallPolicyT;

  constexpr auto threads_per_medium_segment = static_cast<local_segment_index_t>(MediumPolicyT::WARP_THREADS);
  constexpr auto threads_per_small_segment  = static_cast<local_segment_index_t>(SmallPolicyT::WARP_THREADS);

  using MediumAgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, MediumPolicyT, KeyT, ValueT, OffsetT>;

  using SmallAgentWarpMergeSortT =
    sub_warp_merge_sort::AgentSubWarpSort<Order == SortOrder::Descending, SmallPolicyT, KeyT, ValueT, OffsetT>;

  constexpr auto segments_per_medium_block =
    static_cast<local_segment_index_t>(SmallAndMediumPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);

  constexpr auto segments_per_small_block =
    static_cast<local_segment_index_t>(SmallAndMediumPolicyT::SEGMENTS_PER_SMALL_BLOCK);

  __shared__ union
  {
    typename MediumAgentWarpMergeSortT::TempStorage medium_storage[segments_per_medium_block];

    typename SmallAgentWarpMergeSortT::TempStorage small_storage[segments_per_small_block];
  } temp_storage;

  if (bid < medium_blocks)
  {
    const local_segment_index_t sid_within_block  = tid / threads_per_medium_segment;
    const local_segment_index_t medium_segment_id = bid * segments_per_medium_block + sid_within_block;

    if (medium_segment_id < medium_segments)
    {
      const local_segment_index_t global_segment_id = d_medium_segments_indices[medium_segment_id];

      const OffsetT segment_begin = d_begin_offsets[global_segment_id];
      const OffsetT segment_end   = d_end_offsets[global_segment_id];
      const OffsetT num_items     = segment_end - segment_begin;

      MediumAgentWarpMergeSortT(temp_storage.medium_storage[sid_within_block])
        .ProcessSegment(num_items,
                        d_keys_in + segment_begin,
                        d_keys_out + segment_begin,
                        d_values_in + segment_begin,
                        d_values_out + segment_begin);
    }
  }
  else
  {
    const local_segment_index_t sid_within_block = tid / threads_per_small_segment;
    const local_segment_index_t small_segment_id = (bid - medium_blocks) * segments_per_small_block + sid_within_block;

    if (small_segment_id < small_segments)
    {
      const local_segment_index_t global_segment_id = d_small_segments_indices[small_segment_id];

      const OffsetT segment_begin = d_begin_offsets[global_segment_id];
      const OffsetT segment_end   = d_end_offsets[global_segment_id];
      const OffsetT num_items     = segment_end - segment_begin;

      SmallAgentWarpMergeSortT(temp_storage.small_storage[sid_within_block])
        .ProcessSegment(num_items,
                        d_keys_in + segment_begin,
                        d_keys_out + segment_begin,
                        d_values_in + segment_begin,
                        d_values_out + segment_begin);
    }
  }
}

/**
 * @brief Single kernel for large size (more than a few thousand items) segments.
 *
 * @param[in] d_keys_in_orig
 *   Input keys buffer
 *
 * @param[out] d_keys_out_orig
 *   Output keys buffer
 *
 * @param[in] d_values_in_orig
 *   Input values buffer
 *
 * @param[out] d_values_out_orig
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length
 *   `num_segments`, such that `d_begin_offsets[i]` is the first element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length
 *   `num_segments`, such that `d_end_offsets[i]-1` is the last element of the
 *   <em>i</em><sup>th</sup> data segment in `d_keys_*` and `d_values_*`. If
 *   `d_end_offsets[i]-1 <= d_begin_offsets[i]`, the <em>i</em><sup>th</sup> is
 *   considered empty.
 */
template <SortOrder Order,
          typename ChainedPolicyT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT>
__launch_bounds__(ChainedPolicyT::ActivePolicy::LargeSegmentPolicy::BLOCK_THREADS)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortKernelLarge(
    const local_segment_index_t* d_segments_indices,
    const KeyT* d_keys_in_orig,
    KeyT* d_keys_out_orig,
    device_double_buffer<KeyT> d_keys_double_buffer,
    const ValueT* d_values_in_orig,
    ValueT* d_values_out_orig,
    device_double_buffer<ValueT> d_values_double_buffer,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets)
{
  using ActivePolicyT         = typename ChainedPolicyT::ActivePolicy;
  using LargeSegmentPolicyT   = typename ActivePolicyT::LargeSegmentPolicy;
  using local_segment_index_t = local_segment_index_t;

  constexpr int small_tile_size = LargeSegmentPolicyT::BLOCK_THREADS * LargeSegmentPolicyT::ITEMS_PER_THREAD;

  using AgentSegmentedRadixSortT =
    radix_sort::AgentSegmentedRadixSort<Order == SortOrder::Descending, LargeSegmentPolicyT, KeyT, ValueT, OffsetT>;

  __shared__ typename AgentSegmentedRadixSortT::TempStorage storage;

  const local_segment_index_t bid = blockIdx.x;

  constexpr int begin_bit = 0;
  constexpr int end_bit   = sizeof(KeyT) * 8;

  const local_segment_index_t global_segment_id = d_segments_indices[bid];
  const OffsetT segment_begin                   = d_begin_offsets[global_segment_id];
  const OffsetT segment_end                     = d_end_offsets[global_segment_id];
  const OffsetT num_items                       = segment_end - segment_begin;

  constexpr bool keys_only = ::cuda::std::is_same_v<ValueT, NullType>;
  AgentSegmentedRadixSortT agent(num_items, storage);

  d_keys_in_orig += segment_begin;
  d_keys_out_orig += segment_begin;

  if (!keys_only)
  {
    d_values_in_orig += segment_begin;
    d_values_out_orig += segment_begin;
  }

  if (num_items < small_tile_size)
  {
    // Sort in shared memory if the segment fits into it
    agent.ProcessSinglePass(begin_bit, end_bit, d_keys_in_orig, d_values_in_orig, d_keys_out_orig, d_values_out_orig);
  }
  else
  {
    // Sort reading global memory multiple times
    int current_bit = begin_bit;
    int pass_bits   = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

    d_keys_double_buffer = device_double_buffer<KeyT>(
      d_keys_double_buffer.current() + segment_begin, d_keys_double_buffer.alternate() + segment_begin);

    if (!keys_only)
    {
      d_values_double_buffer = device_double_buffer<ValueT>(
        d_values_double_buffer.current() + segment_begin, d_values_double_buffer.alternate() + segment_begin);
    }

    agent.ProcessIterative(
      current_bit,
      pass_bits,
      d_keys_in_orig,
      d_values_in_orig,
      d_keys_double_buffer.current(),
      d_values_double_buffer.current());
    current_bit += pass_bits;

    _CCCL_PRAGMA_NOUNROLL()
    while (current_bit < end_bit)
    {
      pass_bits = (::cuda::std::min) (int{LargeSegmentPolicyT::RADIX_BITS}, (end_bit - current_bit));

      __syncthreads();
      agent.ProcessIterative(
        current_bit,
        pass_bits,
        d_keys_double_buffer.current(),
        d_values_double_buffer.current(),
        d_keys_double_buffer.alternate(),
        d_values_double_buffer.alternate());

      d_keys_double_buffer.swap();
      d_values_double_buffer.swap();
      current_bit += pass_bits;
    }
  }
}
/*
 * Continuation is called after the partitioning stage. It launches kernels
 * to sort large and small segments using the partitioning results. Separation
 * of this stage is required to eliminate device-side synchronization in
 * the CDP mode.
 */
template <typename LargeSegmentPolicyT,
          typename SmallAndMediumPolicyT,
          typename LargeKernelT,
          typename SmallKernelT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN cudaError_t DeviceSegmentedSortContinuation(
  LargeKernelT large_kernel,
  SmallKernelT small_kernel,
  int num_segments,
  KeyT* d_current_keys,
  KeyT* d_final_keys,
  device_double_buffer<KeyT> d_keys_double_buffer,
  ValueT* d_current_values,
  ValueT* d_final_values,
  device_double_buffer<ValueT> d_values_double_buffer,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  local_segment_index_t* group_sizes,
  local_segment_index_t* large_and_medium_segments_indices,
  local_segment_index_t* small_segments_indices,
  cudaStream_t stream)
{
  using local_segment_index_t = local_segment_index_t;

  cudaError error = cudaSuccess;

  const local_segment_index_t large_segments = group_sizes[0];

  if (large_segments > 0)
  {
    // One CTA per segment
    const local_segment_index_t blocks_in_grid = large_segments;

#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking "
            "DeviceSegmentedSortKernelLarge<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(blocks_in_grid),
            LargeSegmentPolicyT::BLOCK_THREADS,
            (long long) stream);
#endif // CUB_DEBUG_LOG

    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(blocks_in_grid, LargeSegmentPolicyT::BLOCK_THREADS, 0, stream)
      .doit(large_kernel,
            large_and_medium_segments_indices,
            d_current_keys,
            d_final_keys,
            d_keys_double_buffer,
            d_current_values,
            d_final_values,
            d_values_double_buffer,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }
  }

  const local_segment_index_t small_segments = group_sizes[1];
  const local_segment_index_t medium_segments =
    static_cast<local_segment_index_t>(num_segments) - (large_segments + small_segments);

  const local_segment_index_t small_blocks =
    ::cuda::ceil_div(small_segments, SmallAndMediumPolicyT::SEGMENTS_PER_SMALL_BLOCK);

  const local_segment_index_t medium_blocks =
    ::cuda::ceil_div(medium_segments, SmallAndMediumPolicyT::SEGMENTS_PER_MEDIUM_BLOCK);

  const local_segment_index_t small_and_medium_blocks_in_grid = small_blocks + medium_blocks;

  if (small_and_medium_blocks_in_grid)
  {
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking "
            "DeviceSegmentedSortKernelSmall<<<%d, %d, 0, %lld>>>()\n",
            static_cast<int>(small_and_medium_blocks_in_grid),
            SmallAndMediumPolicyT::BLOCK_THREADS,
            (long long) stream);
#endif // CUB_DEBUG_LOG

    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
      small_and_medium_blocks_in_grid, SmallAndMediumPolicyT::BLOCK_THREADS, 0, stream)
      .doit(small_kernel,
            small_segments,
            medium_segments,
            medium_blocks,
            small_segments_indices,
            large_and_medium_segments_indices + num_segments - medium_segments,
            d_current_keys,
            d_final_keys,
            d_current_values,
            d_final_values,
            d_begin_offsets,
            d_end_offsets);

    // Check for failure to launch
    error = CubDebug(cudaPeekAtLastError());
    if (cudaSuccess != error)
    {
      return error;
    }

    // Sync the stream if specified to flush runtime errors
    error = CubDebug(DebugSyncStream(stream));
    if (cudaSuccess != error)
    {
      return error;
    }
  }

  return error;
}

#ifdef CUB_RDC_ENABLED
/*
 * Continuation kernel is used only in the CDP mode. It's used to
 * launch DeviceSegmentedSortContinuation as a separate kernel.
 */
template <typename ChainedPolicyT,
          typename LargeKernelT,
          typename SmallKernelT,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT>
__launch_bounds__(1) CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedSortContinuationKernel(
  LargeKernelT large_kernel,
  SmallKernelT small_kernel,
  local_segment_index_t num_segments,
  KeyT* d_current_keys,
  KeyT* d_final_keys,
  device_double_buffer<KeyT> d_keys_double_buffer,
  ValueT* d_current_values,
  ValueT* d_final_values,
  device_double_buffer<ValueT> d_values_double_buffer,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  local_segment_index_t* group_sizes,
  local_segment_index_t* large_and_medium_segments_indices,
  local_segment_index_t* small_segments_indices)
{
  using ActivePolicyT         = typename ChainedPolicyT::ActivePolicy;
  using LargeSegmentPolicyT   = typename ActivePolicyT::LargeSegmentPolicy;
  using SmallAndMediumPolicyT = typename ActivePolicyT::SmallAndMediumSegmentedSortPolicyT;

  // In case of CDP:
  // 1. each CTA has a different main stream
  // 2. all streams are non-blocking
  // 3. child grid always completes before the parent grid
  // 4. streams can be used only from the CTA in which they were created
  // 5. streams created on the host cannot be used on the device
  //
  // Due to (4, 5), we can't pass the user-provided stream in the continuation.
  // Due to (1, 2, 3) it's safe to pass the main stream.
  cudaError_t error =
    detail::segmented_sort::DeviceSegmentedSortContinuation<LargeSegmentPolicyT, SmallAndMediumPolicyT>(
      large_kernel,
      small_kernel,
      num_segments,
      d_current_keys,
      d_final_keys,
      d_keys_double_buffer,
      d_current_values,
      d_final_values,
      d_values_double_buffer,
      d_begin_offsets,
      d_end_offsets,
      group_sizes,
      large_and_medium_segments_indices,
      small_segments_indices,
      0); // always launching on the main stream (see motivation above)

  error = CubDebug(error);
}
#endif // CUB_RDC_ENABLED

} // namespace detail::segmented_sort
CUB_NAMESPACE_END
