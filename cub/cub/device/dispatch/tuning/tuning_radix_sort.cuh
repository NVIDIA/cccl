/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace radix
{
// default
template <std::size_t KeySize, std::size_t ValueSize, std::size_t OffsetSize>
struct sm90_small_key_tuning
{
  static constexpr int threads = 384;
  static constexpr int items   = 23;
};

// clang-format off

// keys
template <> struct sm90_small_key_tuning<1,  0, 4> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<1,  0, 8> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<2,  0, 4> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<2,  0, 8> { static constexpr int threads = 512; static constexpr int items = 19; };

// pairs  8:xx
template <> struct sm90_small_key_tuning<1,  1, 4> { static constexpr int threads = 512; static constexpr int items = 15; };
template <> struct sm90_small_key_tuning<1,  1, 8> { static constexpr int threads = 448; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<1,  2, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<1,  2, 8> { static constexpr int threads = 512; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<1,  4, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<1,  4, 8> { static constexpr int threads = 512; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<1,  8, 4> { static constexpr int threads = 384; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<1,  8, 8> { static constexpr int threads = 384; static constexpr int items = 18; };
template <> struct sm90_small_key_tuning<1, 16, 4> { static constexpr int threads = 512; static constexpr int items = 22; };
template <> struct sm90_small_key_tuning<1, 16, 8> { static constexpr int threads = 512; static constexpr int items = 22; };

// pairs 16:xx
template <> struct sm90_small_key_tuning<2,  1, 4> { static constexpr int threads = 384; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<2,  1, 8> { static constexpr int threads = 384; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<2,  2, 4> { static constexpr int threads = 384; static constexpr int items = 15; };
template <> struct sm90_small_key_tuning<2,  2, 8> { static constexpr int threads = 448; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<2,  4, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<2,  4, 8> { static constexpr int threads = 512; static constexpr int items = 12; };
template <> struct sm90_small_key_tuning<2,  8, 4> { static constexpr int threads = 384; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<2,  8, 8> { static constexpr int threads = 512; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<2, 16, 4> { static constexpr int threads = 512; static constexpr int items = 21; };
template <> struct sm90_small_key_tuning<2, 16, 8> { static constexpr int threads = 576; static constexpr int items = 22; };
// clang-format on

/**
 * @brief Tuning policy for kernel specialization
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
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub
{
  //------------------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------------------

  // Whether this is a keys-only (or key-value) sort
  static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

  // Dominant-sized key/value type
  using DominantT = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------

  /// SM35
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    enum
    {
      PRIMARY_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5, // 1.72B 32b keys/s, 1.17B 32b pairs/s, 1.55B 32b segmented
                                                       // keys/s (K40m)
      ONESWEEP            = false,
      ONESWEEP_RADIX_BITS = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // Scan policy
    using ScanPolicy =
      AgentScanPolicy<1024, 4, OffsetT, BLOCK_LOAD_VECTORIZE, LOAD_DEFAULT, BLOCK_STORE_VECTORIZE, BLOCK_SCAN_WARP_SCANS>;

    // Keys-only downsweep policies
    using DownsweepPolicyKeys = AgentRadixSortDownsweepPolicy<
      128,
      9,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_LDG,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicyKeys = AgentRadixSortDownsweepPolicy<
      64,
      18,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Key-value pairs downsweep policies
    using DownsweepPolicyPairs    = DownsweepPolicyKeys;
    using AltDownsweepPolicyPairs = AgentRadixSortDownsweepPolicy<
      128,
      15,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Downsweep policies
    using DownsweepPolicy = ::cuda::std::_If<KEYS_ONLY, DownsweepPolicyKeys, DownsweepPolicyPairs>;

    using AltDownsweepPolicy = ::cuda::std::_If<KEYS_ONLY, AltDownsweepPolicyKeys, AltDownsweepPolicyPairs>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = DownsweepPolicy;

    // Segmented policies
    using SegmentedPolicy    = DownsweepPolicy;
    using AltSegmentedPolicy = AltDownsweepPolicy;
  };

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy350>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 3.5B 32b keys/s, 1.92B 32b pairs/s (TitanX)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 3.1B 32b segmented keys/s (TitanX)
      ONESWEEP               = false,
      ONESWEEP_RADIX_BITS    = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      160,
      39,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      31,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      256,
      11,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM60 (GP100)
  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 6.9B 32b keys/s (Quadro P100)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 5.9B 32b segmented keys/s (Quadro P100)
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t), // 10.0B 32b keys/s (GP100, 64M random keys)
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      OFFSET_64BIT ? 29 : 30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      25,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      192,
      OFFSET_64BIT ? 32 : 39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM61 (GP104)
  struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 3.4B 32b keys/s, 1.83B 32b pairs/s (1080)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 3.3B 32b segmented keys/s (1080)
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS    = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      384,
      31,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      35,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM62 (Tegra, less RF)
  struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
  {
    enum
    {
      PRIMARY_RADIX_BITS  = 5,
      ALT_RADIX_BITS      = PRIMARY_RADIX_BITS - 1,
      ONESWEEP            = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS = 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      ALT_RADIX_BITS>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy    = DownsweepPolicy;
    using AltSegmentedPolicy = AltDownsweepPolicy;
  };

  /// SM70 (GV100)
  struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5, // 7.62B 32b keys/s (GV100)
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5, // 8.7B 32b segmented keys/s (GV100)
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t), // 15.8B 32b keys/s (V100-SXM2, 64M random keys)
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      256,
      sizeof(KeyT) == 4 && sizeof(ValueT) == 4 ? 46 : 23,
      DominantT,
      4,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      OFFSET_64BIT ? 46 : 47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy =
      AgentRadixSortUpsweepPolicy<256, OFFSET_64BIT ? 46 : 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM80
  struct Policy800 : ChainedPolicy<800, Policy800, Policy700>
  {
    enum
    {
      PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
      SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
      SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
      ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
      ONESWEEP_RADIX_BITS    = 8,
      OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    // Histogram policy
    using HistogramPolicy = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
      384,
      OFFSET_64BIT && sizeof(KeyT) == 4 && !KEYS_ONLY ? 17 : 21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy800>
  {
    static constexpr bool ONESWEEP           = true;
    static constexpr int ONESWEEP_RADIX_BITS = 8;

    using HistogramPolicy    = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

  private:
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5;
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int OFFSET_64BIT           = sizeof(OffsetT) == 8 ? 1 : 0;
    static constexpr int FLOAT_KEYS             = ::cuda::std::is_same<KeyT, float>::value ? 1 : 0;

    using OnesweepPolicyKey32 = AgentRadixSortOnesweepPolicy<
      384,
      KEYS_ONLY ? 20 - OFFSET_64BIT - FLOAT_KEYS
                : (sizeof(ValueT) < 8 ? (OFFSET_64BIT ? 17 : 23) : (OFFSET_64BIT ? 29 : 30)),
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepPolicyKey64 = AgentRadixSortOnesweepPolicy<
      384,
      sizeof(ValueT) < 8 ? 30 : 24,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepLargeKeyPolicy = ::cuda::std::_If<sizeof(KeyT) == 4, OnesweepPolicyKey32, OnesweepPolicyKey64>;

    using OnesweepSmallKeyPolicySizes =
      sm90_small_key_tuning<sizeof(KeyT), KEYS_ONLY ? 0 : sizeof(ValueT), sizeof(OffsetT)>;

    using OnesweepSmallKeyPolicy = AgentRadixSortOnesweepPolicy<
      OnesweepSmallKeyPolicySizes::threads,
      OnesweepSmallKeyPolicySizes::items,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      8>;

  public:
    using OnesweepPolicy = ::cuda::std::_If<sizeof(KeyT) < 4, OnesweepSmallKeyPolicy, OnesweepLargeKeyPolicy>;

    // The Scan, Downsweep and Upsweep policies are never run on SM90, but we have to include them to prevent a
    // compilation error: When we compile e.g. for SM70 **and** SM90, the host compiler will reach calls to those
    // kernels, and instantiate them for MaxPolicy (which is Policy900) on the host, which will reach into the policies
    // below to set the launch bounds. The device compiler pass will also compile all kernels for SM70 **and** SM90,
    // even though only the Onesweep kernel is used on SM90.
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;

    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  using MaxPolicy = Policy900;
};

} // namespace radix
} // namespace detail

// TODO(bgruber): deprecate this alias. Users should not access policy_hubs directly.
/**
 * @brief Tuning policy for kernel specialization
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
template <typename KeyT, typename ValueT, typename OffsetT>
using DeviceRadixSortPolicy = detail::radix::policy_hub<KeyT, ValueT, OffsetT>;

CUB_NAMESPACE_END
