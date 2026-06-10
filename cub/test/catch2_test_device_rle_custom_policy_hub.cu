// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_rle.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/functional>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the rle dispatcher after publishing the tuning API

template <typename LengthT, typename KeyT>
struct my_policy_hub
{
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using RleSweepPolicyT =
      AgentRlePolicy<96,
                     15,
                     BLOCK_LOAD_DIRECT,
                     LOAD_LDG,
                     true,
                     BLOCK_SCAN_WARP_SCANS,
                     cub::detail::default_reduce_by_key_delay_constructor_t<int, int>>;
  };
};

C2H_TEST("DeviceRleDispatch::Dispatch: custom policy hub", "[device][run_length_encode]")
{
  using input_t  = int;
  using offset_t = int;
  using length_t = int;
  using equal_t  = cuda::std::equal_to<>;

  c2h::device_vector<input_t> d_in{1, 1, 2, 2, 2, 3, 3, 4, 4};
  const offset_t num_items = static_cast<offset_t>(d_in.size());

  c2h::device_vector<offset_t> d_offsets(4, thrust::no_init);
  c2h::device_vector<length_t> d_lengths(4, thrust::no_init);
  c2h::device_vector<offset_t> d_num_runs(1);

  const c2h::host_vector<offset_t> expected_offsets{0, 2, 5, 7};
  const c2h::host_vector<length_t> expected_lengths{2, 3, 2, 2};

  using policy_hub_t = my_policy_hub<length_t, input_t>;
  using dispatch_t   = DeviceRleDispatch<input_t*, offset_t*, length_t*, offset_t*, equal_t, offset_t, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_offsets.data()),
    thrust::raw_pointer_cast(d_lengths.data()),
    thrust::raw_pointer_cast(d_num_runs.data()),
    equal_t{},
    num_items,
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_offsets.data()),
    thrust::raw_pointer_cast(d_lengths.data()),
    thrust::raw_pointer_cast(d_num_runs.data()),
    equal_t{},
    num_items,
    /* stream */ nullptr);

  const offset_t num_runs = d_num_runs[0];
  CHECK(num_runs == static_cast<offset_t>(expected_offsets.size()));
  CHECK(d_offsets == expected_offsets);
  CHECK(d_lengths == expected_lengths);
}
