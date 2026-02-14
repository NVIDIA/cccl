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
#include <cub/agent/agent_radix_sort_histogram.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/agent/agent_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>
#include <cub/grid/grid_even_share.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

namespace detail::radix_sort
{
/**
 * @brief Upsweep digit-counting kernel entry point (multi-block).
 *        Computes privatized digit histograms, one per block.
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
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in] d_keys
 *   Input keys buffer
 *
 * @param[out] d_spine
 *   Privatized (per block) digit histograms (striped, i.e., 0s counts from each block,
 *   then 1s counts from each block, etc.)
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] num_bits
 *   Number of bits of current radix digit
 *
 * @param[in] even_share
 *   Even-share descriptor for mapan equal number of tiles onto each thread block
 */
template <typename PolicySelector,
          bool ALT_DIGIT_BITS,
          SortOrder Order,
          typename KeyT,
          typename OffsetT,
          typename DecomposerT = detail::identity_decomposer_t>
__launch_bounds__(int(ALT_DIGIT_BITS ? PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).alt_upsweep.block_threads
                                     : PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).upsweep.block_threads))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortUpsweepKernel(
    const KeyT* d_keys,
    OffsetT* d_spine,
    OffsetT /*num_items*/,
    int current_bit,
    int num_bits,
    GridEvenShare<OffsetT> even_share,
    DecomposerT decomposer = {})
{
  static constexpr radix_sort_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});
  static constexpr radix_sort_upsweep_policy active_upsweep_policy =
    ALT_DIGIT_BITS ? policy.alt_upsweep : policy.upsweep;
  static constexpr radix_sort_downsweep_policy active_downsweep_policy =
    ALT_DIGIT_BITS ? policy.alt_downsweep : policy.downsweep;

  static constexpr int TILE_ITEMS =
    ::cuda::std::max(active_upsweep_policy.block_threads * active_upsweep_policy.items_per_thread,
                     active_downsweep_policy.block_threads * active_downsweep_policy.items_per_thread);

  using ActiveUpsweepPolicyT =
    AgentRadixSortUpsweepPolicy<active_upsweep_policy.block_threads,
                                active_upsweep_policy.items_per_thread,
                                void,
                                active_upsweep_policy.load_modifier,
                                active_upsweep_policy.radix_bits,
                                NoScaling<active_upsweep_policy.block_threads, active_upsweep_policy.items_per_thread>>;

  // Parameterize AgentRadixSortUpsweep type for the current configuration
  using AgentRadixSortUpsweepT =
    detail::radix_sort::AgentRadixSortUpsweep<ActiveUpsweepPolicyT, KeyT, OffsetT, DecomposerT>;

  // Shared memory storage
  __shared__ typename AgentRadixSortUpsweepT::TempStorage temp_storage;

  // Initialize GRID_MAPPING_RAKE even-share descriptor for this thread block
  even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_RAKE>();

  AgentRadixSortUpsweepT upsweep(temp_storage, d_keys, current_bit, num_bits, decomposer);

  upsweep.ProcessRegion(even_share.block_offset, even_share.block_end);

  __syncthreads();

  // Write out digit counts (striped)
  upsweep.template ExtractCounts<Order == SortOrder::Descending>(d_spine, gridDim.x, blockIdx.x);
}

/**
 * @brief Spine scan kernel entry point (single-block).
 *        Computes an exclusive prefix sum over the privatized digit histograms
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @param[in,out] d_spine
 *   Privatized (per block) digit histograms (striped, i.e., 0s counts from each block,
 *   then 1s counts from each block, etc.)
 *
 * @param[in] num_counts
 *   Total number of bin-counts
 */
template <typename PolicySelector, typename OffsetT>
__launch_bounds__(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).scan.block_threads, 1)
  CUB_DETAIL_KERNEL_ATTRIBUTES void RadixSortScanBinsKernel(OffsetT* d_spine, int num_counts)
{
  static constexpr scan_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).scan;
  using ScanPolicy                    = AgentScanPolicy<
                       policy.block_threads,
                       policy.items_per_thread,
                       void,
                       policy.load_algorithm,
                       policy.load_modifier,
                       policy.store_algorithm,
                       policy.scan_algorithm,
                       NoScaling<policy.block_threads, policy.items_per_thread>,
                       delay_constructor_t<policy.delay_constructor.kind,
                                           policy.delay_constructor.delay,
                                           policy.delay_constructor.l2_write_latency>>;

  // Parameterize the AgentScan type for the current configuration
  using AgentScanT = scan::AgentScan<ScanPolicy, OffsetT*, OffsetT*, ::cuda::std::plus<>, OffsetT, OffsetT, OffsetT>;

  // Shared memory storage
  __shared__ typename AgentScanT::TempStorage temp_storage;

  // Block scan instance
  AgentScanT block_scan(temp_storage, d_spine, d_spine, ::cuda::std::plus<>{}, OffsetT(0));

  // Process full input tiles
  int block_offset = 0;
  BlockScanRunningPrefixOp<OffsetT, ::cuda::std::plus<>> prefix_op(0, ::cuda::std::plus<>{});
  while (block_offset + AgentScanT::TILE_ITEMS <= num_counts)
  {
    block_scan.template ConsumeTile<false, false>(block_offset, prefix_op);
    block_offset += AgentScanT::TILE_ITEMS;
  }

  // Process the remaining partial tile (if any).
  if (block_offset < num_counts)
  {
    block_scan.template ConsumeTile<false, true>(block_offset, prefix_op, num_counts - block_offset);
  }
}

/**
 * @brief Downsweep pass kernel entry point (multi-block).
 *        Scatters keys (and values) into corresponding bins for the current digit place.
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
 * @param[in] d_spine
 *   Scan of privatized (per block) digit histograms (striped, i.e., 0s counts from each block,
 *   then 1s counts from each block, etc.)
 *
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] num_bits
 *   Number of bits of current radix digit
 *
 * @param[in] even_share
 *   Even-share descriptor for mapan equal number of tiles onto each thread block
 */
template <typename PolicySelector,
          bool ALT_DIGIT_BITS,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT = detail::identity_decomposer_t>
__launch_bounds__(int(ALT_DIGIT_BITS ? PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).alt_downsweep.block_threads
                                     : PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).downsweep.block_threads))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortDownsweepKernel(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    OffsetT* d_spine,
    OffsetT num_items,
    int current_bit,
    int num_bits,
    GridEvenShare<OffsetT> even_share,
    DecomposerT decomposer = {})
{
  static constexpr radix_sort_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});

  static constexpr radix_sort_upsweep_policy active_upsweep_policy =
    ALT_DIGIT_BITS ? policy.alt_upsweep : policy.upsweep;
  static constexpr radix_sort_downsweep_policy active_downsweep_policy =
    ALT_DIGIT_BITS ? policy.alt_downsweep : policy.downsweep;

  static constexpr int TILE_ITEMS =
    ::cuda::std::max(active_upsweep_policy.block_threads * active_upsweep_policy.items_per_thread,
                     active_downsweep_policy.block_threads * active_downsweep_policy.items_per_thread);

  using ActiveDownsweepPolicyT = AgentRadixSortDownsweepPolicy<
    active_downsweep_policy.block_threads,
    active_downsweep_policy.items_per_thread,
    void,
    active_downsweep_policy.load_algorithm,
    active_downsweep_policy.load_modifier,
    active_downsweep_policy.rank_algorithm,
    active_downsweep_policy.scan_algorithm,
    active_downsweep_policy.radix_bits,
    NoScaling<active_downsweep_policy.block_threads, active_downsweep_policy.items_per_thread>>;

  // Parameterize AgentRadixSortDownsweep type for the current configuration
  using AgentRadixSortDownsweepT = radix_sort::
    AgentRadixSortDownsweep<ActiveDownsweepPolicyT, Order == SortOrder::Descending, KeyT, ValueT, OffsetT, DecomposerT>;

  // Shared memory storage
  __shared__ typename AgentRadixSortDownsweepT::TempStorage temp_storage;

  // Initialize even-share descriptor for this thread block
  even_share.template BlockInit<TILE_ITEMS, GRID_MAPPING_RAKE>();

  // Process input tiles
  AgentRadixSortDownsweepT(
    temp_storage, num_items, d_spine, d_keys_in, d_keys_out, d_values_in, d_values_out, current_bit, num_bits, decomposer)
    .ProcessRegion(even_share.block_offset, even_share.block_end);
}

/**
 * @brief Single pass kernel entry point (single-block).
 *        Fully sorts a tile of input.
 *
 * @tparam SortOrder
 *   Whether or not to use the alternate (lower-bits) policy
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
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
 * @param[in] num_items
 *   Total number of input data items
 *
 * @param[in] current_bit
 *   Bit position of current radix digit
 *
 * @param[in] end_bit
 *   The past-the-end (most-significant) bit index needed for key comparison
 */
template <typename PolicySelector,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
__launch_bounds__(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).single_tile.block_threads, 1)
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortSingleTileKernel(
    const KeyT* d_keys_in,
    KeyT* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    OffsetT num_items,
    int current_bit,
    int end_bit,
    DecomposerT decomposer = {})
{
  // Constants
  static constexpr radix_sort_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10});
  static constexpr int BLOCK_THREADS        = policy.single_tile.block_threads;
  static constexpr int ITEMS_PER_THREAD     = policy.single_tile.items_per_thread;
  static constexpr bool KEYS_ONLY           = ::cuda::std::is_same_v<ValueT, NullType>;

  // BlockRadixSort type
  using BlockRadixSortT =
    BlockRadixSort<KeyT,
                   BLOCK_THREADS,
                   ITEMS_PER_THREAD,
                   ValueT,
                   policy.single_tile.radix_bits,
                   (policy.single_tile.rank_algorithm == RADIX_RANK_MEMOIZE),
                   policy.single_tile.scan_algorithm>;

  // BlockLoad type (keys)
  using BlockLoadKeys = BlockLoad<KeyT, BLOCK_THREADS, ITEMS_PER_THREAD, policy.single_tile.load_algorithm>;

  // BlockLoad type (values)
  using BlockLoadValues = BlockLoad<ValueT, BLOCK_THREADS, ITEMS_PER_THREAD, policy.single_tile.load_algorithm>;

  // Unsigned word for key bits
  using traits           = detail::radix::traits_t<KeyT>;
  using bit_ordered_type = typename traits::bit_ordered_type;

  // Shared memory storage
  __shared__ union TempStorage
  {
    typename BlockRadixSortT::TempStorage sort;
    typename BlockLoadKeys::TempStorage load_keys;
    typename BlockLoadValues::TempStorage load_values;

  } temp_storage;

  // Keys and values for the block
  KeyT keys[ITEMS_PER_THREAD];
  ValueT values[ITEMS_PER_THREAD];

  // Get default (min/max) value for out-of-bounds keys
  bit_ordered_type default_key_bits =
    Order == SortOrder::Descending ? traits::min_raw_binary_key(decomposer) : traits::max_raw_binary_key(decomposer);

  KeyT default_key = reinterpret_cast<KeyT&>(default_key_bits);

  // Load keys
  BlockLoadKeys(temp_storage.load_keys).Load(d_keys_in, keys, num_items, default_key);

  __syncthreads();

  // Load values
  if (!KEYS_ONLY)
  {
    // Register pressure work-around: moving num_items through shfl prevents compiler
    // from reusing guards/addressing from prior guarded loads
    num_items = ShuffleIndex<warp_threads>(num_items, 0, 0xffffffff);

    BlockLoadValues(temp_storage.load_values).Load(d_values_in, values, num_items);

    __syncthreads();
  }

  // Sort tile
  BlockRadixSortT(temp_storage.sort)
    .SortBlockedToStriped(
      keys,
      values,
      current_bit,
      end_bit,
      bool_constant_v < Order == SortOrder::Descending >,
      bool_constant_v<KEYS_ONLY>,
      decomposer);

  // Store keys and values
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    int item_offset = ITEM * BLOCK_THREADS + threadIdx.x;
    if (item_offset < num_items)
    {
      d_keys_out[item_offset] = keys[ITEM];
      if (!KEYS_ONLY)
      {
        d_values_out[item_offset] = values[ITEM];
      }
    }
  }
}

/******************************************************************************
 * Onesweep kernels
 ******************************************************************************/

/**
 * Kernel for computing multiple histograms
 */

/**
 * Histogram kernel
 */
template <typename PolicySelector,
          SortOrder Order,
          typename KeyT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
CUB_DETAIL_KERNEL_ATTRIBUTES __launch_bounds__(
  PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10})
    .histogram.block_threads) void DeviceRadixSortHistogramKernel(OffsetT* d_bins_out,
                                                                  const KeyT* d_keys_in,
                                                                  OffsetT num_items,
                                                                  int start_bit,
                                                                  int end_bit,
                                                                  DecomposerT decomposer = {})
{
  static constexpr radix_sort_histogram_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).histogram;

  using HistogramPolicyT =
    AgentRadixSortHistogramPolicy<policy.block_threads, policy.items_per_thread, policy.num_parts, void, policy.radix_bits>;
  using AgentT = AgentRadixSortHistogram<HistogramPolicyT, Order == SortOrder::Descending, KeyT, OffsetT, DecomposerT>;
  __shared__ typename AgentT::TempStorage temp_storage;
  AgentT agent(temp_storage, d_bins_out, d_keys_in, num_items, start_bit, end_bit, decomposer);
  agent.Process();
}

template <typename PolicySelector,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename PortionOffsetT,
          typename AtomicOffsetT = PortionOffsetT,
          typename DecomposerT   = identity_decomposer_t>
CUB_DETAIL_KERNEL_ATTRIBUTES void
__launch_bounds__(PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).onesweep.block_threads)
  DeviceRadixSortOnesweepKernel(
    AtomicOffsetT* d_lookback,
    AtomicOffsetT* d_ctrs,
    OffsetT* d_bins_out,
    const OffsetT* d_bins_in,
    KeyT* d_keys_out,
    const KeyT* d_keys_in,
    ValueT* d_values_out,
    const ValueT* d_values_in,
    PortionOffsetT num_items,
    int current_bit,
    int num_bits,
    DecomposerT decomposer = {})
{
  static constexpr radix_sort_onesweep_policy policy = PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).onesweep;
  using OnesweepPolicyT                              = AgentRadixSortOnesweepPolicy<
                                 policy.block_threads,
                                 policy.items_per_thread,
                                 void,
                                 policy.rank_num_parts,
                                 policy.rank_algorith,
                                 policy.scan_algorithm,
                                 policy.store_algorithm,
                                 policy.radix_bits,
                                 NoScaling<policy.block_threads, policy.items_per_thread>>;

  using AgentT =
    AgentRadixSortOnesweep<OnesweepPolicyT,
                           Order == SortOrder::Descending,
                           KeyT,
                           ValueT,
                           OffsetT,
                           PortionOffsetT,
                           DecomposerT>;
  __shared__ typename AgentT::TempStorage s;

  AgentT agent(
    s,
    d_lookback,
    d_ctrs,
    d_bins_out,
    d_bins_in,
    d_keys_out,
    d_keys_in,
    d_values_out,
    d_values_in,
    num_items,
    current_bit,
    num_bits,
    decomposer);
  agent.Process();
}

/**
 * Exclusive sum kernel
 */
template <typename PolicySelector, typename OffsetT>
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRadixSortExclusiveSumKernel(OffsetT* d_bins)
{
  static constexpr radix_sort_exclusive_sum_policy policy =
    PolicySelector{}(::cuda::arch_id{CUB_PTX_ARCH / 10}).exclusive_sum;
  constexpr int RADIX_BITS      = policy.radix_bits;
  constexpr int RADIX_DIGITS    = 1 << RADIX_BITS;
  constexpr int BLOCK_THREADS   = policy.block_threads;
  constexpr int BINS_PER_THREAD = (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS;
  using BlockScan               = cub::BlockScan<OffsetT, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // load the bins
  OffsetT bins[BINS_PER_THREAD];
  int bin_start = blockIdx.x * RADIX_DIGITS;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int u = 0; u < BINS_PER_THREAD; ++u)
  {
    int bin = threadIdx.x * BINS_PER_THREAD + u;
    if (bin >= RADIX_DIGITS)
    {
      break;
    }
    bins[u] = d_bins[bin_start + bin];
  }

  // compute offsets
  BlockScan(temp_storage).ExclusiveSum(bins, bins);

  // store the offsets
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int u = 0; u < BINS_PER_THREAD; ++u)
  {
    int bin = threadIdx.x * BINS_PER_THREAD + u;
    if (bin >= RADIX_DIGITS)
    {
      break;
    }
    d_bins[bin_start + bin] = bins[u];
  }
}
} // namespace detail::radix_sort

CUB_NAMESPACE_END
