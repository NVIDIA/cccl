// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * AgentRadixSortDownsweep implements a stateful abstraction of CUDA thread
 * blocks for participating in device-wide radix sort downsweep .
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

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
#  include <cub/agent/agent_radix_sort_upsweep.cuh>
#  include <cub/agent/agent_unique_by_key.cuh>
#endif

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

/**
 * @brief Parameterizable tuning policy type for AgentRadixSortDownsweep
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
 * @tparam LoadAlgorithm
 *   The BlockLoad algorithm to use
 *
 * @tparam LoadModifier
 *   Cache load modifier for reading keys (and values)
 *
 * @tparam RankAlgorithm
 *   The radix ranking algorithm to use
 *
 * @tparam ScanAlgorithm
 *   The block scan algorithm to use
 *
 * @tparam RadixBits
 *   The number of radix bits, i.e., log2(bins)
 */
template <int NominalBlockThreads4B,
          int NominalItemsPerThread4B,
          typename ComputeT,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          RadixRankAlgorithm RankAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          int RadixBits,
          typename ScalingType = detail::RegBoundScaling<NominalBlockThreads4B, NominalItemsPerThread4B, ComputeT>>
struct AgentRadixSortDownsweepPolicy : ScalingType
{
  /// The number of radix bits, i.e., log2(bins)
  static constexpr int RADIX_BITS = RadixBits;

  /// The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = LoadAlgorithm;

  /// Cache load modifier for reading keys (and values)
  static constexpr CacheLoadModifier LOAD_MODIFIER = LoadModifier;

  /// The radix ranking algorithm to use
  static constexpr RadixRankAlgorithm RANK_ALGORITHM = RankAlgorithm;

  /// The BlockScan algorithm to use
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = ScanAlgorithm;
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
namespace detail
{
// Only define this when needed.
// Because of overload woes, this depends on C++20 concepts. util_device.h checks that concepts are available when
// either runtime policies or PTX JSON information are enabled, so if they are, this is always valid. The generic
// version is always defined, and that's the only one needed for regular CUB operations.
//
// TODO: enable this unconditionally once concepts are always available
CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  RadixSortDownsweepAgentPolicy,
  (cub::detail::radix_sort_runtime_policies::RadixSortUpsweepAgentPolicy, UniqueByKeyAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (RADIX_BITS, RadixBits, int),
  (LOAD_ALGORITHM, LoadAlgorithm, cub::BlockLoadAlgorithm),
  (LOAD_MODIFIER, LoadModifier, cub::CacheLoadModifier),
  (RANK_ALGORITHM, RankAlgorithm, cub::RadixRankAlgorithm),
  (SCAN_ALGORITHM, ScanAlgorithm, cub::BlockScanAlgorithm))
} // namespace detail
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

namespace detail::radix_sort
{
/**
 * @brief AgentRadixSortDownsweep implements a stateful abstraction of CUDA thread blocks for participating in
 *        device-wide radix sort downsweep .
 *
 * @tparam AgentRadixSortDownsweepPolicy
 *   Parameterized AgentRadixSortDownsweepPolicy tuning policy type
 *
 * @tparam IS_DESCENDING
 *   Whether or not the sorted-order is high-to-low
 *
 * @tparam KeyT
 *   KeyT type
 *
 * @tparam ValueT
 *   ValueT type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
template <typename AgentRadixSortDownsweepPolicy,
          bool IS_DESCENDING,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
struct AgentRadixSortDownsweep
{
  //---------------------------------------------------------------------
  // Type definitions and constants
  //---------------------------------------------------------------------

  using traits                 = radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = AgentRadixSortDownsweepPolicy::LOAD_ALGORITHM;
  static constexpr CacheLoadModifier LOAD_MODIFIER   = AgentRadixSortDownsweepPolicy::LOAD_MODIFIER;
  static constexpr RadixRankAlgorithm RANK_ALGORITHM = AgentRadixSortDownsweepPolicy::RANK_ALGORITHM;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = AgentRadixSortDownsweepPolicy::SCAN_ALGORITHM;

  static constexpr int BLOCK_THREADS    = AgentRadixSortDownsweepPolicy::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = AgentRadixSortDownsweepPolicy::ITEMS_PER_THREAD;
  static constexpr int RADIX_BITS       = AgentRadixSortDownsweepPolicy::RADIX_BITS;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;

  static constexpr int RADIX_DIGITS = 1 << RADIX_BITS;
  static constexpr bool KEYS_ONLY   = ::cuda::std::is_same_v<ValueT, NullType>;
  static constexpr bool LOAD_WARP_STRIPED =
    RANK_ALGORITHM == RADIX_RANK_MATCH || RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ANY
    || RANK_ALGORITHM == RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR;

  // Input iterator wrapper type (for applying cache modifier)s
  using KeysItr   = CacheModifiedInputIterator<LOAD_MODIFIER, bit_ordered_type, OffsetT>;
  using ValuesItr = CacheModifiedInputIterator<LOAD_MODIFIER, ValueT, OffsetT>;

  // Radix ranking type to use
  using BlockRadixRankT = block_radix_rank_t<RANK_ALGORITHM, BLOCK_THREADS, RADIX_BITS, IS_DESCENDING, SCAN_ALGORITHM>;

  // Digit extractor type
  using fundamental_digit_extractor_t = BFEDigitExtractor<KeyT>;
  using digit_extractor_t = typename traits::template digit_extractor_t<fundamental_digit_extractor_t, DecomposerT>;

  /// Number of bin-starting offsets tracked per thread
  static constexpr int BINS_TRACKED_PER_THREAD = BlockRadixRankT::BINS_TRACKED_PER_THREAD;

  // BlockLoad type (keys)
  using BlockLoadKeysT = BlockLoad<bit_ordered_type, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM>;

  // BlockLoad type (values)
  using BlockLoadValuesT = BlockLoad<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGORITHM>;

  // Value exchange array type
  using ValueExchangeT = ValueT[TILE_ITEMS];

  /**
   * Shared memory storage layout
   */
  union __align__(16) _TempStorage
  {
    typename BlockLoadKeysT::TempStorage load_keys;
    typename BlockLoadValuesT::TempStorage load_values;
    typename BlockRadixRankT::TempStorage radix_rank;

    struct KeysAndOffsets
    {
      bit_ordered_type exchange_keys[TILE_ITEMS];
      OffsetT relative_bin_offsets[RADIX_DIGITS];
    } keys_and_offsets;

    Uninitialized<ValueExchangeT> exchange_values;

    OffsetT exclusive_digit_prefix[RADIX_DIGITS];
  };

  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Thread fields
  //---------------------------------------------------------------------

  // Shared storage for this CTA
  _TempStorage& temp_storage;

  // Input and output device pointers
  KeysItr d_keys_in;
  ValuesItr d_values_in;
  bit_ordered_type* d_keys_out;
  ValueT* d_values_out;

  // The global scatter base offset for each digit (valid in the first RADIX_DIGITS threads)
  OffsetT bin_offset[BINS_TRACKED_PER_THREAD];

  uint32_t current_bit;
  uint32_t num_bits;

  // Whether to short-circuit
  int short_circuit;

  DecomposerT decomposer;

  //---------------------------------------------------------------------
  // Utility methods
  //---------------------------------------------------------------------

  _CCCL_DEVICE _CCCL_FORCEINLINE digit_extractor_t digit_extractor()
  {
    return traits::template digit_extractor<fundamental_digit_extractor_t>(current_bit, num_bits, decomposer);
  }

  /**
   * Scatter ranked keys through shared memory, then to device-accessible memory
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterKeys(
    bit_ordered_type (&twiddled_keys)[ITEMS_PER_THREAD],
    OffsetT (&relative_bin_offsets)[ITEMS_PER_THREAD],
    int (&ranks)[ITEMS_PER_THREAD],
    OffsetT valid_items)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      temp_storage.keys_and_offsets.exchange_keys[ranks[ITEM]] = twiddled_keys[ITEM];
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      bit_ordered_type key       = temp_storage.keys_and_offsets.exchange_keys[threadIdx.x + (ITEM * BLOCK_THREADS)];
      uint32_t digit             = digit_extractor().Digit(key);
      relative_bin_offsets[ITEM] = temp_storage.keys_and_offsets.relative_bin_offsets[digit];

      key = bit_ordered_conversion::from_bit_ordered(decomposer, key);

      if (FULL_TILE || (static_cast<OffsetT>(threadIdx.x + (ITEM * BLOCK_THREADS)) < valid_items))
      {
        d_keys_out[relative_bin_offsets[ITEM] + threadIdx.x + (ITEM * BLOCK_THREADS)] = key;
      }
    }
  }

  /**
   * Scatter ranked values through shared memory, then to device-accessible memory
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterValues(
    ValueT (&values)[ITEMS_PER_THREAD],
    OffsetT (&relative_bin_offsets)[ITEMS_PER_THREAD],
    int (&ranks)[ITEMS_PER_THREAD],
    OffsetT valid_items)
  {
    __syncthreads();

    ValueExchangeT& exchange_values = temp_storage.exchange_values.Alias();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      exchange_values[ranks[ITEM]] = values[ITEM];
    }

    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      ValueT value = exchange_values[threadIdx.x + (ITEM * BLOCK_THREADS)];

      if (FULL_TILE || (static_cast<OffsetT>(threadIdx.x + (ITEM * BLOCK_THREADS)) < valid_items))
      {
        d_values_out[relative_bin_offsets[ITEM] + threadIdx.x + (ITEM * BLOCK_THREADS)] = value;
      }
    }
  }

  /**
   * Load a tile of keys (specialized for full tile, block load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadKeys(
    bit_ordered_type (&keys)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    bit_ordered_type oob_item,
    ::cuda::std::true_type is_full_tile,
    ::cuda::std::false_type warp_striped)
  {
    BlockLoadKeysT(temp_storage.load_keys).Load(d_keys_in + block_offset, keys);

    __syncthreads();
  }

  /**
   * Load a tile of keys (specialized for partial tile, block load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadKeys(
    bit_ordered_type (&keys)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    bit_ordered_type oob_item,
    ::cuda::std::false_type is_full_tile,
    ::cuda::std::false_type warp_striped)
  {
    // Register pressure work-around: moving valid_items through shfl prevents compiler
    // from reusing guards/addressing from prior guarded loads
    valid_items = ShuffleIndex<warp_threads>(valid_items, 0, 0xffffffff);

    BlockLoadKeysT(temp_storage.load_keys).Load(d_keys_in + block_offset, keys, valid_items, oob_item);

    __syncthreads();
  }

  /**
   * Load a tile of keys (specialized for full tile, warp-striped load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadKeys(
    bit_ordered_type (&keys)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    bit_ordered_type oob_item,
    ::cuda::std::true_type is_full_tile,
    ::cuda::std::true_type warp_striped)
  {
    LoadDirectWarpStriped(threadIdx.x, d_keys_in + block_offset, keys);
  }

  /**
   * Load a tile of keys (specialized for partial tile, warp-striped load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadKeys(
    bit_ordered_type (&keys)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    bit_ordered_type oob_item,
    ::cuda::std::false_type is_full_tile,
    ::cuda::std::true_type warp_striped)
  {
    // Register pressure work-around: moving valid_items through shfl prevents compiler
    // from reusing guards/addressing from prior guarded loads
    valid_items = ShuffleIndex<warp_threads>(valid_items, 0, 0xffffffff);

    LoadDirectWarpStriped(threadIdx.x, d_keys_in + block_offset, keys, valid_items, oob_item);
  }

  /**
   * Load a tile of values (specialized for full tile, block load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadValues(
    ValueT (&values)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    ::cuda::std::true_type is_full_tile,
    ::cuda::std::false_type warp_striped)
  {
    BlockLoadValuesT(temp_storage.load_values).Load(d_values_in + block_offset, values);

    __syncthreads();
  }

  /**
   * Load a tile of values (specialized for partial tile, block load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadValues(
    ValueT (&values)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    ::cuda::std::false_type is_full_tile,
    ::cuda::std::false_type warp_striped)
  {
    // Register pressure work-around: moving valid_items through shfl prevents compiler
    // from reusing guards/addressing from prior guarded loads
    valid_items = ShuffleIndex<warp_threads>(valid_items, 0, 0xffffffff);

    BlockLoadValuesT(temp_storage.load_values).Load(d_values_in + block_offset, values, valid_items);

    __syncthreads();
  }

  /**
   * Load a tile of items (specialized for full tile, warp-striped load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadValues(
    ValueT (&values)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    ::cuda::std::true_type is_full_tile,
    ::cuda::std::true_type warp_striped)
  {
    LoadDirectWarpStriped(threadIdx.x, d_values_in + block_offset, values);
  }

  /**
   * Load a tile of items (specialized for partial tile, warp-striped load)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadValues(
    ValueT (&values)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    ::cuda::std::false_type is_full_tile,
    ::cuda::std::true_type warp_striped)
  {
    // Register pressure work-around: moving valid_items through shfl prevents compiler
    // from reusing guards/addressing from prior guarded loads
    valid_items = ShuffleIndex<warp_threads>(valid_items, 0, 0xffffffff);

    LoadDirectWarpStriped(threadIdx.x, d_values_in + block_offset, values, valid_items);
  }

  /**
   * Truck along associated values
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void GatherScatterValues(
    OffsetT (&relative_bin_offsets)[ITEMS_PER_THREAD],
    int (&ranks)[ITEMS_PER_THREAD],
    OffsetT block_offset,
    OffsetT valid_items,
    ::cuda::std::false_type /*is_keys_only*/)
  {
    ValueT values[ITEMS_PER_THREAD];

    __syncthreads();

    LoadValues(values, block_offset, valid_items, bool_constant_v<FULL_TILE>, bool_constant_v<LOAD_WARP_STRIPED>);

    ScatterValues<FULL_TILE>(values, relative_bin_offsets, ranks, valid_items);
  }

  /**
   * Truck along associated values (specialized for key-only sorting)
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void GatherScatterValues(
    OffsetT (& /*relative_bin_offsets*/)[ITEMS_PER_THREAD],
    int (& /*ranks*/)[ITEMS_PER_THREAD],
    OffsetT /*block_offset*/,
    OffsetT /*valid_items*/,
    ::cuda::std::true_type /*is_keys_only*/)
  {}

  /**
   * Process tile
   */
  template <bool FULL_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessTile(OffsetT block_offset, OffsetT valid_items = TILE_ITEMS)
  {
    bit_ordered_type keys[ITEMS_PER_THREAD];
    int ranks[ITEMS_PER_THREAD];
    OffsetT relative_bin_offsets[ITEMS_PER_THREAD];

    // Assign default (min/max) value to all keys
    bit_ordered_type default_key =
      IS_DESCENDING ? traits::min_raw_binary_key(decomposer) : traits::max_raw_binary_key(decomposer);

    // Load tile of keys
    LoadKeys(
      keys, block_offset, valid_items, default_key, bool_constant_v<FULL_TILE>, bool_constant_v<LOAD_WARP_STRIPED>);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int KEY = 0; KEY < ITEMS_PER_THREAD; KEY++)
    {
      keys[KEY] = bit_ordered_conversion::to_bit_ordered(decomposer, keys[KEY]);
    }

    // Rank the twiddled keys
    int exclusive_digit_prefix[BINS_TRACKED_PER_THREAD];
    BlockRadixRankT(temp_storage.radix_rank).RankKeys(keys, ranks, digit_extractor(), exclusive_digit_prefix);

    __syncthreads();

    // Share exclusive digit prefix
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;
      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        // Store exclusive prefix
        temp_storage.exclusive_digit_prefix[bin_idx] = exclusive_digit_prefix[track];
      }
    }

    __syncthreads();

    // Get inclusive digit prefix
    int inclusive_digit_prefix[BINS_TRACKED_PER_THREAD];

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;
      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        if (IS_DESCENDING)
        {
          // Get inclusive digit prefix from exclusive prefix (higher bins come first)
          inclusive_digit_prefix[track] =
            (bin_idx == 0) ? (BLOCK_THREADS * ITEMS_PER_THREAD) : temp_storage.exclusive_digit_prefix[bin_idx - 1];
        }
        else
        {
          // Get inclusive digit prefix from exclusive prefix (lower bins come first)
          inclusive_digit_prefix[track] =
            (bin_idx == RADIX_DIGITS - 1)
              ? (BLOCK_THREADS * ITEMS_PER_THREAD)
              : temp_storage.exclusive_digit_prefix[bin_idx + 1];
        }
      }
    }

    __syncthreads();

    // Update global scatter base offsets for each digit
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;
      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        bin_offset[track] -= exclusive_digit_prefix[track];
        temp_storage.keys_and_offsets.relative_bin_offsets[bin_idx] = bin_offset[track];
        bin_offset[track] += inclusive_digit_prefix[track];
      }
    }

    __syncthreads();

    // Scatter keys
    ScatterKeys<FULL_TILE>(keys, relative_bin_offsets, ranks, valid_items);

    // Gather/scatter values
    GatherScatterValues<FULL_TILE>(relative_bin_offsets, ranks, block_offset, valid_items, bool_constant_v<KEYS_ONLY>);
  }

  //---------------------------------------------------------------------
  // Copy shortcut
  //---------------------------------------------------------------------

  /**
   * Copy tiles within the range of input
   */
  template <typename InputIteratorT, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Copy(InputIteratorT d_in, T* d_out, OffsetT block_offset, OffsetT block_end)
  {
    // Simply copy the input
    while (block_end - block_offset >= TILE_ITEMS)
    {
      T items[ITEMS_PER_THREAD];

      LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in + block_offset, items);
      __syncthreads();
      StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);

      block_offset += TILE_ITEMS;
    }

    // Clean up last partial tile with guarded-I/O
    if (block_offset < block_end)
    {
      OffsetT valid_items = block_end - block_offset;

      T items[ITEMS_PER_THREAD];

      LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_in + block_offset, items, valid_items);
      __syncthreads();
      StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items, valid_items);
    }
  }

  /**
   * Copy tiles within the range of input (specialized for NullType)
   */
  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  Copy(InputIteratorT /*d_in*/, NullType* /*d_out*/, OffsetT /*block_offset*/, OffsetT /*block_end*/)
  {}

  //---------------------------------------------------------------------
  // Interface
  //---------------------------------------------------------------------

  /**
   * Constructor
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentRadixSortDownsweep(
    TempStorage& temp_storage,
    OffsetT (&bin_offset)[BINS_TRACKED_PER_THREAD],
    OffsetT num_items,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    int current_bit,
    int num_bits,
    DecomposerT decomposer = {})
      : temp_storage(temp_storage.Alias())
      , d_keys_in(reinterpret_cast<const bit_ordered_type*>(d_keys_in))
      , d_values_in(d_values_in)
      , d_keys_out(reinterpret_cast<bit_ordered_type*>(d_keys_out))
      , d_values_out(d_values_out)
      , current_bit(current_bit)
      , num_bits(num_bits)
      , short_circuit(1)
      , decomposer(decomposer)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      this->bin_offset[track] = bin_offset[track];

      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;
      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        // Short circuit if the histogram has only bin counts of only zeros or problem-size
        short_circuit = short_circuit && ((bin_offset[track] == 0) || (bin_offset[track] == num_items));
      }
    }

    short_circuit = __syncthreads_and(short_circuit);
  }

  /**
   * Constructor
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentRadixSortDownsweep(
    TempStorage& temp_storage,
    OffsetT num_items,
    OffsetT* d_spine,
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    int current_bit,
    int num_bits,
    DecomposerT decomposer = {})
      : temp_storage(temp_storage.Alias())
      , d_keys_in(reinterpret_cast<const bit_ordered_type*>(d_keys_in))
      , d_values_in(d_values_in)
      , d_keys_out(reinterpret_cast<bit_ordered_type*>(d_keys_out))
      , d_values_out(d_values_out)
      , current_bit(current_bit)
      , num_bits(num_bits)
      , short_circuit(1)
      , decomposer(decomposer)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
    {
      int bin_idx = (threadIdx.x * BINS_TRACKED_PER_THREAD) + track;

      // Load digit bin offsets (each of the first RADIX_DIGITS threads will load an offset for that digit)
      if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
      {
        if (IS_DESCENDING)
        {
          bin_idx = RADIX_DIGITS - bin_idx - 1;
        }

        // Short circuit if the first block's histogram has only bin counts of only zeros or problem-size
        OffsetT first_block_bin_offset = d_spine[gridDim.x * bin_idx];
        short_circuit = short_circuit && ((first_block_bin_offset == 0) || (first_block_bin_offset == num_items));

        // Load my block's bin offset for my bin
        bin_offset[track] = d_spine[(gridDim.x * bin_idx) + blockIdx.x];
      }
    }

    short_circuit = __syncthreads_and(short_circuit);
  }

  /**
   * Distribute keys from a segment of input tiles.
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ProcessRegion(OffsetT block_offset, OffsetT block_end)
  {
    if (short_circuit)
    {
      // Copy keys
      Copy(d_keys_in, d_keys_out, block_offset, block_end);

      // Copy values
      Copy(d_values_in, d_values_out, block_offset, block_end);
    }
    else
    {
      // Process full tiles of tile_items
      _CCCL_PRAGMA_NOUNROLL()
      while (block_end - block_offset >= TILE_ITEMS)
      {
        ProcessTile<true>(block_offset);
        block_offset += TILE_ITEMS;

        __syncthreads();
      }

      // Clean up last partial tile with guarded-I/O
      if (block_offset < block_end)
      {
        ProcessTile<false>(block_offset, block_end - block_offset);
      }
    }
  }
};
} // namespace detail::radix_sort

CUB_NAMESPACE_END
