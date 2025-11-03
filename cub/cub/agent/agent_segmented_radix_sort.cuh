// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cub/block/block_radix_sort.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
/**
 * This agent will be implementing the `DeviceSegmentedRadixSort` when the
 * https://github.com/NVIDIA/cub/issues/383 is addressed.
 *
 * @tparam IsDescending
 *   Whether or not the sorted-order is high-to-low
 *
 * @tparam SegmentedPolicyT
 *   Chained tuning policy
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <bool IsDescending,
          typename SegmentedPolicyT,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
struct AgentSegmentedRadixSort
{
  OffsetT num_items;

  static constexpr int ITEMS_PER_THREAD = SegmentedPolicyT::ITEMS_PER_THREAD;
  static constexpr int BLOCK_THREADS    = SegmentedPolicyT::BLOCK_THREADS;
  static constexpr int RADIX_BITS       = SegmentedPolicyT::RADIX_BITS;
  static constexpr int RADIX_DIGITS     = 1 << RADIX_BITS;
  static constexpr int KEYS_ONLY        = ::cuda::std::is_same_v<ValueT, NullType>;

  using traits           = radix::traits_t<KeyT>;
  using bit_ordered_type = typename traits::bit_ordered_type;

  // Huge segment handlers
  using BlockUpsweepT   = AgentRadixSortUpsweep<SegmentedPolicyT, KeyT, OffsetT, DecomposerT>;
  using DigitScanT      = BlockScan<OffsetT, BLOCK_THREADS>;
  using BlockDownsweepT = AgentRadixSortDownsweep<SegmentedPolicyT, IsDescending, KeyT, ValueT, OffsetT, DecomposerT>;

  /// Number of bin-starting offsets tracked per thread
  static constexpr int BINS_TRACKED_PER_THREAD = BlockDownsweepT::BINS_TRACKED_PER_THREAD;

  // Small segment handlers
  using BlockRadixSortT =
    BlockRadixSort<KeyT,
                   BLOCK_THREADS,
                   ITEMS_PER_THREAD,
                   ValueT,
                   RADIX_BITS,
                   (SegmentedPolicyT::RANK_ALGORITHM == RADIX_RANK_MEMOIZE),
                   SegmentedPolicyT::SCAN_ALGORITHM>;

  using BlockKeyLoadT = BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, SegmentedPolicyT::LOAD_ALGORITHM>;

  using BlockValueLoadT = BlockLoad<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, SegmentedPolicyT::LOAD_ALGORITHM>;

  union _TempStorage
  {
    // Huge segment handlers
    typename BlockUpsweepT::TempStorage upsweep;
    typename BlockDownsweepT::TempStorage downsweep;

    struct UnboundBlockSort
    {
      OffsetT reverse_counts_in[RADIX_DIGITS];
      OffsetT reverse_counts_out[RADIX_DIGITS];
      typename DigitScanT::TempStorage scan;
    } unbound_sort;

    // Small segment handlers
    typename BlockKeyLoadT::TempStorage keys_load;
    typename BlockValueLoadT::TempStorage values_load;
    typename BlockRadixSortT::TempStorage sort;
  };

  using TempStorage = Uninitialized<_TempStorage>;
  _TempStorage& temp_storage;

  DecomposerT decomposer;

  _CCCL_DEVICE _CCCL_FORCEINLINE
  AgentSegmentedRadixSort(OffsetT num_items, TempStorage& temp_storage, DecomposerT decomposer = {})
      : num_items(num_items)
      , temp_storage(temp_storage.Alias())
      , decomposer(decomposer)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessSinglePass(
    int begin_bit, int end_bit, const KeyT* d_keys_in, const ValueT* d_values_in, KeyT* d_keys_out, ValueT* d_values_out)
  {
    KeyT thread_keys[ITEMS_PER_THREAD];
    ValueT thread_values[ITEMS_PER_THREAD];

    // For FP64 the difference is:
    // Lowest() -> -1.79769e+308 = 00...00b -> TwiddleIn -> -0 = 10...00b
    // LOWEST   -> -nan          = 11...11b -> TwiddleIn ->  0 = 00...00b

    bit_ordered_type default_key_bits =
      IsDescending ? traits::min_raw_binary_key(decomposer) : traits::max_raw_binary_key(decomposer);
    KeyT oob_default = reinterpret_cast<KeyT&>(default_key_bits);

    if (!KEYS_ONLY)
    {
      BlockValueLoadT(temp_storage.values_load).Load(d_values_in, thread_values, num_items);

      __syncthreads();
    }

    {
      BlockKeyLoadT(temp_storage.keys_load).Load(d_keys_in, thread_keys, num_items, oob_default);

      __syncthreads();
    }

    BlockRadixSortT(temp_storage.sort)
      .SortBlockedToStriped(
        thread_keys,
        thread_values,
        begin_bit,
        end_bit,
        bool_constant_v<IsDescending>,
        bool_constant_v<KEYS_ONLY>,
        decomposer);

    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out, thread_keys, num_items);

    if (!KEYS_ONLY)
    {
      cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out, thread_values, num_items);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessIterative(
    int current_bit,
    int pass_bits,
    const KeyT* d_keys_in,
    const ValueT* d_values_in,
    KeyT* d_keys_out,
    ValueT* d_values_out)
  {
    // Upsweep
    BlockUpsweepT upsweep(temp_storage.upsweep, d_keys_in, current_bit, pass_bits, decomposer);
    upsweep.ProcessRegion(OffsetT{}, num_items);

    __syncthreads();

    // The count of each digit value in this pass (valid in the first RADIX_DIGITS threads)
    OffsetT bin_count[BINS_TRACKED_PER_THREAD];
    upsweep.ExtractCounts(bin_count);

    __syncthreads();

    if (IsDescending)
    {
      // Reverse bin counts
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          temp_storage.unbound_sort.reverse_counts_in[bin_idx] = bin_count[track];
        }
      }

      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          bin_count[track] = temp_storage.unbound_sort.reverse_counts_in[RADIX_DIGITS - bin_idx - 1];
        }
      }
    }

    // Scan
    // The global scatter base offset for each digit value in this pass
    // (valid in the first RADIX_DIGITS threads)
    OffsetT bin_offset[BINS_TRACKED_PER_THREAD];
    DigitScanT(temp_storage.unbound_sort.scan).ExclusiveSum(bin_count, bin_offset);

    if (IsDescending)
    {
      // Reverse bin offsets
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          temp_storage.unbound_sort.reverse_counts_out[threadIdx.x] = bin_offset[track];
        }
      }

      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
      {
        int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

        if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
        {
          bin_offset[track] = temp_storage.unbound_sort.reverse_counts_out[RADIX_DIGITS - bin_idx - 1];
        }
      }
    }

    __syncthreads();

    // Downsweep
    BlockDownsweepT downsweep(
      temp_storage.downsweep,
      bin_offset,
      num_items,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      current_bit,
      pass_bits,
      decomposer);
    downsweep.ProcessRegion(OffsetT{}, num_items);
  }
};
} // namespace detail::radix_sort

CUB_NAMESPACE_END
