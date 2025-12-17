// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * AgentRadixSortUpsweep implements a stateful abstraction of CUDA thread blocks for participating in device-wide radix
 * sort upsweep .
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_reduce.cuh>

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
#  include <cub/agent/agent_radix_sort_histogram.cuh>
#endif

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * @brief Parameterizable tuning policy type for AgentRadixSortUpsweep
 *
 * @tparam NominalBlockThreads4B
 *   Threads per thread block
 *
 * @tparam NominalItemsPerThread4B
 *   Items per thread (per tile of input)
 *
 * @tparam ComputeT
 *   Dominant compute type
 *
 * @tparam LoadModifier
 *   Cache load modifier for reading keys
 *
 * @tparam RadixBits
 *   The number of radix bits, i.e., log2(bins)
 */
template <int NominalBlockThreads4B,
          int NominalItemsPerThread4B,
          typename ComputeT,
          CacheLoadModifier LoadModifier,
          int RadixBits,
          typename ScalingType = detail::RegBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>>
struct AgentRadixSortUpsweepPolicy : ScalingType
{
  /// The number of radix bits, i.e., log2(bins)
  static constexpr int RADIX_BITS = RadixBits;

  /// Cache load modifier for reading keys
  static constexpr CacheLoadModifier LOAD_MODIFIER = LoadModifier;
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
namespace detail::radix_sort_runtime_policies
{
// Only define this when needed.
// Because of overload woes, this depends on C++20 concepts. util_device.h checks that concepts are available when
// either runtime policies or PTX JSON information are enabled, so if they are, this is always valid. The generic
// version is always defined, and that's the only one needed for regular CUB operations.
//
// TODO: enable this unconditionally once concepts are always available
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  RadixSortUpsweepAgentPolicy,
  (GenericAgentPolicy, RadixSortExclusiveSumAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (RADIX_BITS, RadixBits, int),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier))
} // namespace detail::radix_sort_runtime_policies
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail::radix_sort
{
/**
 * @brief AgentRadixSortUpsweep implements a stateful abstraction of CUDA thread blocks for
 * participating in device-wide radix sort upsweep .
 *
 * @tparam AgentRadixSortUpsweepPolicy
 *   Parameterized AgentRadixSortUpsweepPolicy tuning policy type
 *
 * @tparam KeyT
 *   KeyT type
 *
 * @tparam DecomposerT = identity_decomposer_t
 *   Signed integer type for global offsets
 */
template <typename AgentRadixSortUpsweepPolicy,
          typename KeyT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
struct AgentRadixSortUpsweep
{
  //---------------------------------------------------------------------
  // Type definitions and constants
  //---------------------------------------------------------------------
  using traits                 = radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

  // Integer type for digit counters (to be packed into words of PackedCounters)
  using DigitCounter = unsigned char;

  // Integer type for packing DigitCounters into columns of shared memory banks
  using PackedCounter = unsigned int;

  static constexpr CacheLoadModifier LOAD_MODIFIER = AgentRadixSortUpsweepPolicy::LOAD_MODIFIER;

  static constexpr int RADIX_BITS      = AgentRadixSortUpsweepPolicy::RADIX_BITS;
  static constexpr int BLOCK_THREADS   = AgentRadixSortUpsweepPolicy::BLOCK_THREADS;
  static constexpr int KEYS_PER_THREAD = AgentRadixSortUpsweepPolicy::ITEMS_PER_THREAD;

  static constexpr int RADIX_DIGITS = 1 << RADIX_BITS;

  static constexpr int LOG_WARP_THREADS = log2_warp_threads;
  static constexpr int WARP_THREADS     = 1 << LOG_WARP_THREADS;
  static constexpr int WARPS            = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS;

  static constexpr int TILE_ITEMS = BLOCK_THREADS * KEYS_PER_THREAD;

  static constexpr int BYTES_PER_COUNTER    = sizeof(DigitCounter);
  static constexpr int OG_BYTES_PER_COUNTER = Log2<BYTES_PER_COUNTER>::VALUE;

  static constexpr int PACKING_RATIO     = sizeof(PackedCounter) / sizeof(DigitCounter);
  static constexpr int LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE;

  static constexpr int LOG_COUNTER_LANES = ::cuda::std::max(0, int(RADIX_BITS) - int(LOG_PACKING_RATIO));
  static constexpr int COUNTER_LANES     = 1 << LOG_COUNTER_LANES;

  // To prevent counter overflow, we must periodically unpack and aggregate the
  // digit counters back into registers.  Each counter lane is assigned to a
  // warp for aggregation.

  static constexpr int LANES_PER_WARP = ::cuda::std::max(1, (COUNTER_LANES + WARPS - 1) / WARPS);

  // Unroll tiles in batches without risk of counter overflow
  static constexpr int UNROLL_COUNT      = ::cuda::std::min(64, 255 / KEYS_PER_THREAD);
  static constexpr int UNROLLED_ELEMENTS = UNROLL_COUNT * TILE_ITEMS;

  // Input iterator wrapper type (for applying cache modifier)s
  using KeysItr = CacheModifiedInputIterator<LOAD_MODIFIER, bit_ordered_type, OffsetT>;

  // Digit extractor type
  using fundamental_digit_extractor_t = BFEDigitExtractor<KeyT>;
  using digit_extractor_t = typename traits::template digit_extractor_t<fundamental_digit_extractor_t, DecomposerT>;

  /**
   * Shared memory storage layout
   */
  union __align__(16) _TempStorage
  {
    DigitCounter thread_counters[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
    PackedCounter packed_thread_counters[COUNTER_LANES][BLOCK_THREADS];
    OffsetT block_counters[WARP_THREADS][RADIX_DIGITS];
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Thread fields (aggregate state bundle)
  //---------------------------------------------------------------------

  // Shared storage for this CTA
  _TempStorage& temp_storage;

  // Thread-local counters for periodically aggregating composite-counter lanes
  OffsetT local_counts[LANES_PER_WARP][PACKING_RATIO];

  // Input and output device pointers
  KeysItr d_keys_in;

  // Target bits
  int current_bit;
  int num_bits;
  DecomposerT decomposer;

  //---------------------------------------------------------------------
  // Helper structure for templated iteration
  //---------------------------------------------------------------------

  // Iterate
  template <int COUNT, int MAX>
  struct Iterate
  {
    // BucketKeys
    static _CCCL_DEVICE _CCCL_FORCEINLINE void
    BucketKeys(AgentRadixSortUpsweep& cta, bit_ordered_type keys[KEYS_PER_THREAD])
    {
      cta.Bucket(keys[COUNT]);

      // Next
      Iterate<COUNT + 1, MAX>::BucketKeys(cta, keys);
    }
  };

  // Terminate
  template <int MAX>
  struct Iterate<MAX, MAX>
  {
    // BucketKeys
    static _CCCL_DEVICE _CCCL_FORCEINLINE void
    BucketKeys(AgentRadixSortUpsweep& /*cta*/, bit_ordered_type /*keys*/[KEYS_PER_THREAD])
    {}
  };

  //---------------------------------------------------------------------
  // Utility methods
  //---------------------------------------------------------------------
  _CCCL_DEVICE _CCCL_FORCEINLINE digit_extractor_t digit_extractor()
  {
    return traits::template digit_extractor<fundamental_digit_extractor_t>(current_bit, num_bits, decomposer);
  }

  /**
   * Decode a key and increment corresponding smem digit counter
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void Bucket(bit_ordered_type key)
  {
    // Perform transform op
    bit_ordered_type converted_key = bit_ordered_conversion::to_bit_ordered(decomposer, key);

    // Extract current digit bits
    uint32_t digit = digit_extractor().Digit(converted_key);

    // Get sub-counter offset
    uint32_t sub_counter = digit & (PACKING_RATIO - 1);

    // Get row offset
    uint32_t row_offset = digit >> LOG_PACKING_RATIO;

    // Increment counter
    temp_storage.thread_counters[row_offset][threadIdx.x][sub_counter]++;
  }

  /**
   * Reset composite counters
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ResetDigitCounters()
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int LANE = 0; LANE < COUNTER_LANES; LANE++)
    {
      temp_storage.packed_thread_counters[LANE][threadIdx.x] = 0;
    }
  }

  /**
   * Reset the unpacked counters in each thread
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ResetUnpackedCounters()
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
      {
        local_counts[LANE][UNPACKED_COUNTER] = 0;
      }
    }
  }

  /**
   * Extracts and aggregates the digit counters for each counter lane
   * owned by this warp
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void UnpackDigitCounts()
  {
    unsigned int warp_id  = threadIdx.x >> LOG_WARP_THREADS;
    unsigned int warp_tid = ::cuda::ptx::get_sreg_laneid();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
    {
      const int counter_lane = (LANE * WARPS) + warp_id;
      if (counter_lane < COUNTER_LANES)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int PACKED_COUNTER = 0; PACKED_COUNTER < BLOCK_THREADS; PACKED_COUNTER += WARP_THREADS)
        {
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
          {
            OffsetT counter = temp_storage.thread_counters[counter_lane][warp_tid + PACKED_COUNTER][UNPACKED_COUNTER];
            local_counts[LANE][UNPACKED_COUNTER] += counter;
          }
        }
      }
    }
  }

  /**
   * Processes a single, full tile
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessFullTile(OffsetT block_offset)
  {
    // Tile of keys
    bit_ordered_type keys[KEYS_PER_THREAD];

    LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_in + block_offset, keys);

    // Prevent hoisting
    __syncthreads();

    // Bucket tile of keys
    Iterate<0, KEYS_PER_THREAD>::BucketKeys(*this, keys);
  }

  /**
   * Processes a single load (may have some threads masked off)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessPartialTile(OffsetT block_offset, const OffsetT& block_end)
  {
    // Process partial tile if necessary using single loads
    for (OffsetT offset = threadIdx.x; offset < block_end - block_offset; offset += BLOCK_THREADS)
    {
      // Load and bucket key
      bit_ordered_type key = d_keys_in[block_offset + offset];
      Bucket(key);
    }
  }

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /**
   * Constructor
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentRadixSortUpsweep(
    TempStorage& temp_storage, const KeyT* d_keys_in, int current_bit, int num_bits, DecomposerT decomposer = {})
      : temp_storage(temp_storage.Alias())
      , d_keys_in(reinterpret_cast<const bit_ordered_type*>(d_keys_in))
      , current_bit(current_bit)
      , num_bits(num_bits)
      , decomposer(decomposer)
  {}

  /**
   * Compute radix digit histograms from a segment of input tiles.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessRegion(OffsetT block_offset, const OffsetT& block_end)
  {
    // Reset digit counters in smem and unpacked counters in registers
    ResetDigitCounters();
    ResetUnpackedCounters();

    // Unroll batches of full tiles
    while (block_end - block_offset >= UNROLLED_ELEMENTS)
    {
      for (int i = 0; i < UNROLL_COUNT; ++i)
      {
        ProcessFullTile(block_offset);
        block_offset += TILE_ITEMS;
      }

      __syncthreads();

      // Aggregate back into local_count registers to prevent overflow
      UnpackDigitCounts();

      __syncthreads();

      // Reset composite counters in lanes
      ResetDigitCounters();
    }

    // Unroll single full tiles
    while (block_end - block_offset >= TILE_ITEMS)
    {
      ProcessFullTile(block_offset);
      block_offset += TILE_ITEMS;
    }

    // Process partial tile if necessary
    ProcessPartialTile(block_offset, block_end);

    __syncthreads();

    // Aggregate back into local_count registers
    UnpackDigitCounts();
  }

  /**
   * Extract counts (saving them to the external array)
   */
  template <bool IS_DESCENDING>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExtractCounts(OffsetT* counters, int bin_stride = 1, int bin_offset = 0)
  {
    unsigned int warp_id  = threadIdx.x >> LOG_WARP_THREADS;
    unsigned int warp_tid = ::cuda::ptx::get_sreg_laneid();

    // Place unpacked digit counters in shared memory
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
    {
      int counter_lane = (LANE * WARPS) + warp_id;
      if (counter_lane < COUNTER_LANES)
      {
        int digit_row = counter_lane << LOG_PACKING_RATIO;

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
        {
          int bin_idx = digit_row + UNPACKED_COUNTER;

          temp_storage.block_counters[warp_tid][bin_idx] = local_counts[LANE][UNPACKED_COUNTER];
        }
      }
    }

    __syncthreads();

    // Rake-reduce bin_count reductions

    // Whole blocks
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int BIN_BASE = RADIX_DIGITS % BLOCK_THREADS; (BIN_BASE + BLOCK_THREADS) <= RADIX_DIGITS;
         BIN_BASE += BLOCK_THREADS)
    {
      int bin_idx       = BIN_BASE + threadIdx.x;
      OffsetT bin_count = 0;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < WARP_THREADS; ++i)
      {
        bin_count += temp_storage.block_counters[i][bin_idx];
      }

      if (IS_DESCENDING)
      {
        bin_idx = RADIX_DIGITS - bin_idx - 1;
      }

      counters[(bin_stride * bin_idx) + bin_offset] = bin_count;
    }

    // Remainder
    if ((RADIX_DIGITS % BLOCK_THREADS != 0) && (threadIdx.x < RADIX_DIGITS))
    {
      int bin_idx       = threadIdx.x;
      OffsetT bin_count = 0;

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < WARP_THREADS; ++i)
      {
        bin_count += temp_storage.block_counters[i][bin_idx];
      }

      if (IS_DESCENDING)
      {
        bin_idx = RADIX_DIGITS - bin_idx - 1;
      }

      counters[(bin_stride * bin_idx) + bin_offset] = bin_count;
    }
  }

  /**
   * @brief Extract counts
   *
   * @param[out] bin_count
   *   The exclusive prefix sum for the digits
   *   [(threadIdx.x * BINS_TRACKED_PER_THREAD) ... (threadIdx.x * BINS_TRACKED_PER_THREAD) + BINS_TRACKED_PER_THREAD -
   * 1]
   */
  template <int BINS_TRACKED_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ExtractCounts(OffsetT (&bin_count)[BINS_TRACKED_PER_THREAD])
  {
    unsigned int warp_id  = threadIdx.x >> LOG_WARP_THREADS;
    unsigned int warp_tid = ::cuda::ptx::get_sreg_laneid();

    // Place unpacked digit counters in shared memory
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int LANE = 0; LANE < LANES_PER_WARP; LANE++)
    {
      int counter_lane = (LANE * WARPS) + warp_id;
      if (counter_lane < COUNTER_LANES)
      {
        int digit_row = counter_lane << LOG_PACKING_RATIO;

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO; UNPACKED_COUNTER++)
        {
          int bin_idx = digit_row + UNPACKED_COUNTER;

          temp_storage.block_counters[warp_tid][bin_idx] = local_counts[LANE][UNPACKED_COUNTER];
        }
      }
    }

    __syncthreads();

    // Rake-reduce bin_count reductions
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        bin_count[track] = 0;

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int i = 0; i < WARP_THREADS; ++i)
        {
          bin_count[track] += temp_storage.block_counters[i][bin_idx];
        }
      }
    }
  }
};
} // namespace detail::radix_sort

CUB_NAMESPACE_END
