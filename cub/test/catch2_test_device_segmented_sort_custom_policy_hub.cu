// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <algorithm>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the segmented sort dispatcher after publishing the tuning
// API

template <typename KeyT>
struct my_policy_hub
{
  // from Policy500 of the CUB segmented sort tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = 6;
    static constexpr int PARTITIONING_THRESHOLD = 300;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      16,
      KeyT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_BITS>;

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  4 /* Threads per segment */,
                                  7 /* items per thread */,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  7 /* items per thread */,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
  };
};

C2H_TEST("DispatchSegmentedSort::Dispatch: custom policy hub", "[segmented][sort][device]")
{
  using key_t                    = int;
  using value_t                  = NullType;
  using offset_t                 = int;
  constexpr bool is_overwrite_ok = false;

  c2h::host_vector<key_t> h_keys_in{7, 2, 5, 1, 4, 3, 9, 8, 6, 0};
  c2h::host_vector<offset_t> h_offsets{0, 4, 7, 10};

  c2h::device_vector<key_t> d_keys_in = h_keys_in;
  c2h::device_vector<key_t> d_keys_out(h_keys_in.size(), thrust::no_init);
  c2h::device_vector<offset_t> d_offsets = h_offsets;

  DoubleBuffer<key_t> d_keys(thrust::raw_pointer_cast(d_keys_in.data()), thrust::raw_pointer_cast(d_keys_out.data()));
  DoubleBuffer<value_t> d_values;

  const auto num_items    = static_cast<::cuda::std::int64_t>(h_keys_in.size());
  const auto num_segments = static_cast<offset_t>(h_offsets.size() - 1);

  using policy_hub_t = my_policy_hub<key_t>;
  using dispatch_t =
    DispatchSegmentedSort<SortOrder::Ascending, key_t, value_t, offset_t, const offset_t*, const offset_t*, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    d_keys,
    d_values,
    num_items,
    num_segments,
    thrust::raw_pointer_cast(d_offsets.data()),
    thrust::raw_pointer_cast(d_offsets.data()) + 1,
    is_overwrite_ok,
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    d_keys,
    d_values,
    num_items,
    num_segments,
    thrust::raw_pointer_cast(d_offsets.data()),
    thrust::raw_pointer_cast(d_offsets.data()) + 1,
    is_overwrite_ok,
    /* stream */ nullptr);

  c2h::host_vector<key_t> expected = h_keys_in;
  for (std::size_t seg = 0; seg < h_offsets.size() - 1; ++seg)
  {
    const auto begin = expected.begin() + h_offsets[seg];
    const auto end   = expected.begin() + h_offsets[seg + 1];
    std::stable_sort(begin, end);
  }

  const auto& d_sorted = (d_keys.selector == 0) ? d_keys_in : d_keys_out;
  REQUIRE(d_sorted == expected);
}
