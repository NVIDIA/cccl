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

#include <cub/agent/agent_radix_sort_downsweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector, bool AltDigitBits>
_CCCL_API constexpr int segmented_radix_sort_kernel_launch_bounds()
{
  constexpr auto policy = current_policy<PolicySelector>();
  return AltDigitBits ? policy.alt_segmented.block_threads : policy.segmented.block_threads;
}

/**
 * @brief Segmented radix sorting pass (one block per segment)
 *
 * @tparam PolicySelector
 *   Selects the tuning policy
 *
 * @tparam AltDigitBits
 *   Whether or not to use the alternate (lower-bits) policy
 *
 * @tparam SortOrder
 *   Whether to sort in ascending or descending order
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys_in
 *   Input keys buffer
 *
 * @param[in] d_keys_out
 *   Output keys buffer
 *
 * @param[in] d_values_in
 *   Input values buffer
 *
 * @param[in] d_values_out
 *   Output values buffer
 *
 * @param[in] d_begin_offsets
 *   Random-access input iterator to the sequence of beginning offsets of length `num_segments`,
 *   such that <tt>d_begin_offsets[i]</tt> is the first element of the <em>i</em><sup>th</sup>
 *   data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>
 *
 * @param[in] d_end_offsets
 *   Random-access input iterator to the sequence of ending offsets of length `num_segments`,
 *   such that <tt>d_end_offsets[i]-1</tt> is the last element of the <em>i</em><sup>th</sup>
 *   data segment in <tt>d_keys_*</tt> and <tt>d_values_*</tt>.
 *   If <tt>d_end_offsets[i]-1</tt> <= <tt>d_begin_offsets[i]</tt>,
 *   the <em>i</em><sup>th</sup> is considered empty.
 *
 * @param[in] num_segments
 *   The number of segments that comprise the sorting data
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] pass_bits
 *   Number of bits of current radix digit
 */
template <typename PolicySelector,
          bool AltDigitBits,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SegmentSizeT,
          typename DecomposerT = detail::identity_decomposer_t>
#if _CCCL_HAS_CONCEPTS()
  requires radix_sort_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(segmented_radix_sort_kernel_launch_bounds<PolicySelector, AltDigitBits>())
  _CCCL_KERNEL_ATTRIBUTES void DeviceSegmentedRadixSortKernel(
    _CCCL_GRID_CONSTANT const KeyT* const d_keys_in,
    _CCCL_GRID_CONSTANT KeyT* const d_keys_out,
    _CCCL_GRID_CONSTANT const ValueT* const d_values_in,
    _CCCL_GRID_CONSTANT ValueT* const d_values_out,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorT d_begin_offsets,
    _CCCL_GRID_CONSTANT const EndOffsetIteratorT d_end_offsets,
    _CCCL_GRID_CONSTANT const int current_bit,
    _CCCL_GRID_CONSTANT const int pass_bits,
    _CCCL_GRID_CONSTANT const DecomposerT decomposer = {})
{
  //
  // Constants
  //

  static constexpr radix_sort_policy policy                  = current_policy<PolicySelector>();
  static constexpr radix_sort_downsweep_policy active_policy = AltDigitBits ? policy.alt_segmented : policy.segmented;

  static constexpr int block_threads = active_policy.block_threads;
  static constexpr int radix_bits    = active_policy.radix_bits;
  static constexpr int radix_digits  = 1 << radix_bits;

  using ActiveUpsweepPolicyT =
    AgentRadixSortUpsweepPolicy<active_policy.block_threads,
                                active_policy.items_per_thread,
                                void,
                                active_policy.load_modifier,
                                active_policy.radix_bits,
                                NoScaling<active_policy.block_threads, active_policy.items_per_thread>>;

  using BlockUpsweepT = AgentRadixSortUpsweep<ActiveUpsweepPolicyT, KeyT, SegmentSizeT, DecomposerT>;

  using DigitScanT = BlockScan<SegmentSizeT, block_threads>;

  using ActiveDownsweepPolicyT = AgentRadixSortDownsweepPolicy<
    active_policy.block_threads,
    active_policy.items_per_thread,
    void,
    active_policy.load_algorithm,
    active_policy.load_modifier,
    active_policy.rank_algorithm,
    active_policy.scan_algorithm,
    active_policy.radix_bits,
    NoScaling<active_policy.block_threads, active_policy.items_per_thread>>;

  using BlockDownsweepT =
    AgentRadixSortDownsweep<ActiveDownsweepPolicyT, Order == SortOrder::Descending, KeyT, ValueT, SegmentSizeT, DecomposerT>;

  /// Number of bin-starting offsets tracked per thread
  static constexpr int bins_tracked_per_thread = BlockDownsweepT::BINS_TRACKED_PER_THREAD;

  //
  // Process input tiles
  //

  // Shared memory storage
  __shared__ union
  {
    typename BlockUpsweepT::TempStorage upsweep;
    typename BlockDownsweepT::TempStorage downsweep;
    struct
    {
      volatile SegmentSizeT reverse_counts_in[radix_digits];
      volatile SegmentSizeT reverse_counts_out[radix_digits];
      typename DigitScanT::TempStorage scan;
    };

  } temp_storage;

  const auto segment_id = blockIdx.x;

  // Ensure the size of the current segment does not overflow SegmentSizeT
  _CCCL_ASSERT(static_cast<decltype(d_end_offsets[segment_id] - d_begin_offsets[segment_id])>(
                 ::cuda::std::numeric_limits<SegmentSizeT>::max())
                 > (d_end_offsets[segment_id] - d_begin_offsets[segment_id]),
               "A single segment size is limited to the maximum value representable by SegmentSizeT");
  const auto num_items = static_cast<SegmentSizeT>(d_end_offsets[segment_id] - d_begin_offsets[segment_id]);

  // Check if empty segment
  if (num_items <= 0)
  {
    return;
  }

  // Upsweep
  BlockUpsweepT upsweep(
    temp_storage.upsweep, d_keys_in + d_begin_offsets[segment_id], current_bit, pass_bits, decomposer);
  upsweep.ProcessRegion(SegmentSizeT{0}, num_items);

  __syncthreads();

  // The count of each digit value in this pass (valid in the first RADIX_DIGITS threads)
  SegmentSizeT bin_count[bins_tracked_per_thread];
  upsweep.ExtractCounts(bin_count);

  __syncthreads();

  if (Order == SortOrder::Descending)
  {
    // Reverse bin counts
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < bins_tracked_per_thread; ++track)
    {
      int bin_idx = (threadIdx.x * bins_tracked_per_thread) + track;

      if ((block_threads == radix_digits) || (bin_idx < radix_digits))
      {
        temp_storage.reverse_counts_in[bin_idx] = bin_count[track];
      }
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < bins_tracked_per_thread; ++track)
    {
      int bin_idx = (threadIdx.x * bins_tracked_per_thread) + track;

      if ((block_threads == radix_digits) || (bin_idx < radix_digits))
      {
        bin_count[track] = temp_storage.reverse_counts_in[radix_digits - bin_idx - 1];
      }
    }
  }

  // Scan
  SegmentSizeT bin_offset[bins_tracked_per_thread]; // The scatter base offset within the segment for each digit value
                                                    // in this pass (valid in the first RADIX_DIGITS threads)
  DigitScanT(temp_storage.scan).ExclusiveSum(bin_count, bin_offset);

  if (Order == SortOrder::Descending)
  {
    // Reverse bin offsets
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < bins_tracked_per_thread; ++track)
    {
      int bin_idx = (threadIdx.x * bins_tracked_per_thread) + track;

      if ((block_threads == radix_digits) || (bin_idx < radix_digits))
      {
        temp_storage.reverse_counts_out[threadIdx.x] = bin_offset[track];
      }
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < bins_tracked_per_thread; ++track)
    {
      int bin_idx = (threadIdx.x * bins_tracked_per_thread) + track;

      if ((block_threads == radix_digits) || (bin_idx < radix_digits))
      {
        bin_offset[track] = temp_storage.reverse_counts_out[radix_digits - bin_idx - 1];
      }
    }
  }

  __syncthreads();

  // Downsweep
  BlockDownsweepT downsweep(
    temp_storage.downsweep,
    bin_offset,
    num_items,
    d_keys_in + d_begin_offsets[segment_id],
    d_keys_out + d_begin_offsets[segment_id],
    d_values_in + d_begin_offsets[segment_id],
    d_values_out + d_begin_offsets[segment_id],
    current_bit,
    pass_bits,
    decomposer);
  downsweep.ProcessRegion(SegmentSizeT{0}, num_items);
}
} // namespace detail::radix_sort

CUB_NAMESPACE_END
