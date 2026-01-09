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

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
/**
 * @brief Segmented radix sorting pass (one block per segment)
 *
 * @tparam ChainedPolicyT
 *   Chained tuning policy
 *
 * @tparam ALT_DIGIT_BITS
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
template <typename ChainedPolicyT,
          bool ALT_DIGIT_BITS,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename SegmentSizeT,
          typename DecomposerT = detail::identity_decomposer_t>
__launch_bounds__(int((ALT_DIGIT_BITS) ? ChainedPolicyT::ActivePolicy::AltSegmentedPolicy::BLOCK_THREADS
                                       : ChainedPolicyT::ActivePolicy::SegmentedPolicy::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedRadixSortKernel(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int current_bit,
    int pass_bits,
    DecomposerT decomposer = {})
{
  //
  // Constants
  //

  using SegmentedPolicyT =
    ::cuda::std::_If<ALT_DIGIT_BITS,
                     typename ChainedPolicyT::ActivePolicy::AltSegmentedPolicy,
                     typename ChainedPolicyT::ActivePolicy::SegmentedPolicy>;

  static constexpr int BLOCK_THREADS = SegmentedPolicyT::BLOCK_THREADS;
  static constexpr int RADIX_BITS    = SegmentedPolicyT::RADIX_BITS;
  static constexpr int RADIX_DIGITS  = 1 << RADIX_BITS;

  // Upsweep type
  using BlockUpsweepT = detail::radix_sort::AgentRadixSortUpsweep<SegmentedPolicyT, KeyT, SegmentSizeT, DecomposerT>;

  // Digit-scan type
  using DigitScanT = BlockScan<SegmentSizeT, BLOCK_THREADS>;

  // Downsweep type
  using BlockDownsweepT = detail::radix_sort::
    AgentRadixSortDownsweep<SegmentedPolicyT, Order == SortOrder::Descending, KeyT, ValueT, SegmentSizeT, DecomposerT>;

  /// Number of bin-starting offsets tracked per thread
  static constexpr int BINS_TRACKED_PER_THREAD = BlockDownsweepT::BINS_TRACKED_PER_THREAD;

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
      volatile SegmentSizeT reverse_counts_in[RADIX_DIGITS];
      volatile SegmentSizeT reverse_counts_out[RADIX_DIGITS];
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
  SegmentSizeT bin_count[BINS_TRACKED_PER_THREAD];
  upsweep.ExtractCounts(bin_count);

  __syncthreads();

  if (Order == SortOrder::Descending)
  {
    // Reverse bin counts
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        temp_storage.reverse_counts_in[bin_idx] = bin_count[track];
      }
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        bin_count[track] = temp_storage.reverse_counts_in[RADIX_DIGITS - bin_idx - 1];
      }
    }
  }

  // Scan
  SegmentSizeT bin_offset[BINS_TRACKED_PER_THREAD]; // The scatter base offset within the segment for each digit value
                                                    // in this pass (valid in the first RADIX_DIGITS threads)
  DigitScanT(temp_storage.scan).ExclusiveSum(bin_count, bin_offset);

  if (Order == SortOrder::Descending)
  {
    // Reverse bin offsets
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        temp_storage.reverse_counts_out[threadIdx.x] = bin_offset[track];
      }
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        bin_offset[track] = temp_storage.reverse_counts_out[RADIX_DIGITS - bin_idx - 1];
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
