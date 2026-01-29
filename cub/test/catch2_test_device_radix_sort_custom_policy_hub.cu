// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>

#include "catch2_radix_sort_helper.cuh"

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API

template <typename KeyT, typename OffsetT>
struct my_policy_hub
{
  using DominantT = KeyT;

  // from Policy500 of the CUB radix sort tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5;
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr bool ONESWEEP              = false;
    static constexpr int ONESWEEP_RADIX_BITS    = 8;

    using HistogramPolicy    = AgentRadixSortHistogramPolicy<256, 8, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;
    using OnesweepPolicy     = AgentRadixSortOnesweepPolicy<
          256,
          21,
          DominantT,
          1,
          RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
          BLOCK_SCAN_WARP_SCANS,
          RADIX_SORT_STORE_DIRECT,
          ONESWEEP_RADIX_BITS>;
    using ScanPolicy =
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;
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
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;
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
};

C2H_TEST("DispatchRadixSort::Dispatch: custom policy hub", "[keys][radix][sort][device]")
{
  using key_t              = int;
  using offset_t           = unsigned;
  const offset_t num_items = 12345;

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::gen(C2H_SEED(1), in_keys);

  DoubleBuffer<key_t> d_keys(thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));
  DoubleBuffer<NullType> d_values;

  using policy_hub_t = my_policy_hub<key_t, offset_t>;
  using dispatch_t =
    DispatchRadixSort<SortOrder::Ascending, key_t, NullType, offset_t, cub::detail::identity_decomposer_t, policy_hub_t>;
  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys,
    d_values,
    num_items,
    0,
    sizeof(key_t) * CHAR_BIT,
    /* is_overwrite_ok */ false,
    /* stream */ nullptr);
  c2h::device_vector<uint8_t> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    d_keys,
    d_values,
    num_items,
    0,
    sizeof(key_t) * CHAR_BIT,
    /* is_overwrite_ok */ false,
    /* stream */ nullptr);

  const auto ref_keys = radix_sort_reference(in_keys, /* is_decending */ false);
  REQUIRE(ref_keys == out_keys);
}
