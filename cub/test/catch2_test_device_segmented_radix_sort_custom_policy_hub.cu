// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_radix_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include "catch2_radix_sort_helper.cuh"
#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the segmented radix sort dispatcher after publishing the
// tuning API

template <typename KeyT, typename OffsetT>
struct my_policy_hub
{
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
          KeyT,
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
      KeyT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      KeyT,
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
      KeyT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      31,
      KeyT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
      256,
      11,
      KeyT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };
};

C2H_TEST("DispatchSegmentedRadixSort::Dispatch: custom policy hub", "[keys][segmented][radix][sort][device]")
{
  using key_t          = int;
  using value_t        = NullType;
  using segment_size_t = int;
  using offset_t       = int;

  c2h::device_vector<key_t> in_keys(1234);
  c2h::gen(C2H_SEED(1), in_keys);
  c2h::device_vector<key_t> out_keys(in_keys);
  c2h::device_vector<offset_t> offsets{0, 40, 777, 1002};

  DoubleBuffer<key_t> d_keys(thrust::raw_pointer_cast(in_keys.data()), thrust::raw_pointer_cast(out_keys.data()));
  DoubleBuffer<value_t> d_values;

  const auto num_items    = static_cast<::cuda::std::int64_t>(in_keys.size());
  const auto num_segments = static_cast<::cuda::std::int64_t>(offsets.size() - 1);

  using policy_hub_t = my_policy_hub<key_t, segment_size_t>;
  using dispatch_t   = DispatchSegmentedRadixSort<
      SortOrder::Ascending,
      key_t,
      value_t,
      const offset_t*,
      const offset_t*,
      segment_size_t,
      policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys,
    d_values,
    num_items,
    num_segments,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    begin_bit<key_t>(),
    end_bit<key_t>(),
    /* is_overwrite_okay */ false,
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    d_keys,
    d_values,
    num_items,
    num_segments,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    begin_bit<key_t>(),
    end_bit<key_t>(),
    /* is_overwrite_okay */ false,
    /* stream */ nullptr);

  const auto ref_keys  = segmented_radix_sort_reference(in_keys, /* is_descending */ false, offsets);
  const auto& d_sorted = (d_keys.selector == 0) ? in_keys : out_keys;
  REQUIRE(d_sorted == ref_keys);
}
