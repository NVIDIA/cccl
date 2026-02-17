// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge_sort.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <algorithm>

#include "catch2_test_device_merge_sort_common.cuh"
#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the merge sort dispatcher after publishing the tuning API

template <typename KeyIteratorT>
struct my_policy_hub
{
  using KeyT = cub::detail::it_value_t<KeyIteratorT>;

  // from Policy500 of the CUB merge sort tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(11),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };
};

C2H_TEST("DispatchMergeSort::Dispatch: custom policy hub", "[merge][sort][device]")
{
  using key_t              = int;
  using offset_t           = unsigned;
  const offset_t num_items = 12345;

  c2h::device_vector<key_t> in_keys(num_items);
  c2h::device_vector<key_t> out_keys(num_items);
  c2h::gen(C2H_SEED(1), in_keys);

  using policy_hub_t = my_policy_hub<key_t*>;
  using dispatch_t = DispatchMergeSort<key_t*, NullType*, key_t*, NullType*, offset_t, custom_less_op_t, policy_hub_t>;
  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(in_keys.data()),
    nullptr,
    thrust::raw_pointer_cast(out_keys.data()),
    nullptr,
    num_items,
    custom_less_op_t{},
    /* stream */ nullptr);
  c2h::device_vector<uint8_t> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(in_keys.data()),
    nullptr,
    thrust::raw_pointer_cast(out_keys.data()),
    nullptr,
    num_items,
    custom_less_op_t{},
    /* stream */ nullptr);

  c2h::host_vector<key_t> ref_keys = in_keys;
  std::stable_sort(ref_keys.begin(), ref_keys.end(), custom_less_op_t{});
  REQUIRE(ref_keys == out_keys);
}
