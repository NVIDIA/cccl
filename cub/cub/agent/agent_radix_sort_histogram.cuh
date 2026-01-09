// SPDX-FileCopyrightText: Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * agent_radix_sort_histogram.cuh implements a stateful abstraction of CUDA
 * thread blocks for participating in the device histogram kernel used for
 * one-sweep radix sorting.
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
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/operations.h>

CUB_NAMESPACE_BEGIN

template <int BlockThreads, int ItemsPerThread, int NOMINAL_4B_NUM_PARTS, typename ComputeT, int RadixBits>
struct AgentRadixSortHistogramPolicy
{
  static constexpr int BLOCK_THREADS    = BlockThreads;
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
  /** NUM_PARTS is the number of private histograms (parts) each histogram is split
   * into. Each warp lane is assigned to a specific part based on the lane
   * ID. However, lanes with the same ID in different warp use the same private
   * histogram. This arrangement helps reduce the degree of conflicts in atomic
   * operations. */
  static constexpr int NUM_PARTS =
    ::cuda::std::max(1, NOMINAL_4B_NUM_PARTS * 4 / ::cuda::std::max(int{sizeof(ComputeT)}, 4));
  static constexpr int RADIX_BITS = RadixBits;
};

template <int BlockThreads, int RadixBits>
struct AgentRadixSortExclusiveSumPolicy
{
  static constexpr int BLOCK_THREADS = BlockThreads;
  static constexpr int RADIX_BITS    = RadixBits;
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
  RadixSortExclusiveSumAgentPolicy, (always_true), (BLOCK_THREADS, BlockThreads, int), (RADIX_BITS, RadixBits, int) )

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  RadixSortHistogramAgentPolicy,
  (GenericAgentPolicy, RadixSortExclusiveSumAgentPolicy),
  (BLOCK_THREADS, BlockThreads, int),
  (ITEMS_PER_THREAD, ItemsPerThread, int),
  (NUM_PARTS, NumParts, int),
  (RADIX_BITS, RadixBits, int) )
} // namespace detail::radix_sort_runtime_policies
#endif // defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)

namespace detail::radix_sort
{
template <typename AgentRadixSortHistogramPolicy,
          bool IS_DESCENDING,
          typename KeyT,
          typename OffsetT,
          typename DecomposerT = identity_decomposer_t>
struct AgentRadixSortHistogram
{
  // constants
  static constexpr int ITEMS_PER_THREAD = AgentRadixSortHistogramPolicy::ITEMS_PER_THREAD;
  static constexpr int BLOCK_THREADS    = AgentRadixSortHistogramPolicy::BLOCK_THREADS;
  static constexpr int TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int RADIX_BITS       = AgentRadixSortHistogramPolicy::RADIX_BITS;
  static constexpr int RADIX_DIGITS     = 1 << RADIX_BITS;
  static constexpr int MAX_NUM_PASSES   = (sizeof(KeyT) * 8 + RADIX_BITS - 1) / RADIX_BITS;
  static constexpr int NUM_PARTS        = AgentRadixSortHistogramPolicy::NUM_PARTS;

  using traits                 = radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

  using Twiddle             = RadixSortTwiddle<IS_DESCENDING, KeyT>;
  using ShmemCounterT       = uint32_t;
  using ShmemAtomicCounterT = ShmemCounterT;

  using fundamental_digit_extractor_t = ShiftDigitExtractor<KeyT>;
  using digit_extractor_t = typename traits::template digit_extractor_t<fundamental_digit_extractor_t, DecomposerT>;

  struct _TempStorage
  {
    ShmemAtomicCounterT bins[MAX_NUM_PASSES][RADIX_DIGITS][NUM_PARTS];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  // thread fields
  // shared memory storage
  _TempStorage& s;

  // bins for the histogram
  OffsetT* d_bins_out;

  // data to compute the histogram
  const bit_ordered_type* d_keys_in;

  // number of data items
  OffsetT num_items;

  // begin and end bits for sorting
  int begin_bit, end_bit;

  // number of sorting passes
  int num_passes;

  DecomposerT decomposer;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentRadixSortHistogram(
    TempStorage& temp_storage,
    OffsetT* d_bins_out,
    const KeyT* d_keys_in,
    OffsetT num_items,
    int begin_bit,
    int end_bit,
    DecomposerT decomposer = {})
      : s(temp_storage.Alias())
      , d_bins_out(d_bins_out)
      , d_keys_in(reinterpret_cast<const bit_ordered_type*>(d_keys_in))
      , num_items(num_items)
      , begin_bit(begin_bit)
      , end_bit(end_bit)
      , num_passes((end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS)
      , decomposer(decomposer)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void Init()
  {
    // Initialize bins to 0.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int bin = threadIdx.x; bin < RADIX_DIGITS; bin += BLOCK_THREADS)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int pass = 0; pass < num_passes; ++pass)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int part = 0; part < NUM_PARTS; ++part)
        {
          s.bins[pass][bin][part] = 0;
        }
      }
    }
    __syncthreads();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void LoadTileKeys(OffsetT tile_offset, bit_ordered_type (&keys)[ITEMS_PER_THREAD])
  {
    // tile_offset < num_items always, hence the line below works
    bool full_tile = num_items - tile_offset >= TILE_ITEMS;
    if (full_tile)
    {
      LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_in + tile_offset, keys);
    }
    else
    {
      LoadDirectStriped<BLOCK_THREADS>(
        threadIdx.x, d_keys_in + tile_offset, keys, num_items - tile_offset, Twiddle::DefaultKey(decomposer));
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int u = 0; u < ITEMS_PER_THREAD; ++u)
    {
      keys[u] = Twiddle::In(keys[u], decomposer);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  AccumulateSharedHistograms(OffsetT tile_offset, bit_ordered_type (&keys)[ITEMS_PER_THREAD])
  {
    int part = ::cuda::ptx::get_sreg_laneid() % NUM_PARTS;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int current_bit = begin_bit, pass = 0; current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
    {
      const int num_bits = ::cuda::std::min(+RADIX_BITS, end_bit - current_bit);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int u = 0; u < ITEMS_PER_THREAD; ++u)
      {
        uint32_t bin = digit_extractor(current_bit, num_bits).Digit(keys[u]);
        // Using cuda::atomic<> results in lower performance on GP100,
        // so atomicAdd() is used instead.
        atomicAdd(&s.bins[pass][bin][part], 1);
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void AccumulateGlobalHistograms()
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int bin = threadIdx.x; bin < RADIX_DIGITS; bin += BLOCK_THREADS)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int pass = 0; pass < num_passes; ++pass)
      {
        OffsetT count = cub::ThreadReduce(s.bins[pass][bin], ::cuda::std::plus<>{});
        if (count > 0)
        {
          // Using cuda::atomic<> here would also require using it in
          // other kernels. However, other kernels of onesweep sorting
          // (ExclusiveSum, Onesweep) don't need atomic
          // access. Therefore, atomicAdd() is used, until
          // cuda::atomic_ref<> becomes available.
          atomicAdd(&d_bins_out[pass * RADIX_DIGITS + bin], count);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    // Within a portion, avoid overflowing (u)int32 counters.
    // Between portions, accumulate results in global memory.
    constexpr OffsetT MAX_PORTION_SIZE = 1 << 30;
    OffsetT num_portions               = ::cuda::ceil_div(num_items, MAX_PORTION_SIZE);
    for (OffsetT portion = 0; portion < num_portions; ++portion)
    {
      // Reset the counters.
      Init();
      __syncthreads();

      // Process the tiles.
      OffsetT portion_offset = portion * MAX_PORTION_SIZE;
      OffsetT portion_size   = ::cuda::std::min(MAX_PORTION_SIZE, num_items - portion_offset);
      for (OffsetT offset = blockIdx.x * TILE_ITEMS; offset < portion_size; offset += TILE_ITEMS * gridDim.x)
      {
        OffsetT tile_offset = portion_offset + offset;
        bit_ordered_type keys[ITEMS_PER_THREAD];
        LoadTileKeys(tile_offset, keys);
        AccumulateSharedHistograms(tile_offset, keys);
      }
      __syncthreads();

      // Accumulate the result in global memory.
      AccumulateGlobalHistograms();
      __syncthreads();
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE digit_extractor_t digit_extractor(int current_bit, int num_bits)
  {
    return traits::template digit_extractor<fundamental_digit_extractor_t>(current_bit, num_bits, decomposer);
  }
};
} // namespace detail::radix_sort

CUB_NAMESPACE_END
