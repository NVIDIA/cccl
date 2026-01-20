// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/functional>

#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the scan dispatcher after publishing the tuning API

template <typename InputValueT, typename OutputValueT, typename AccumT, typename OffsetT, typename ScanOpT>
struct my_policy_hub
{
  // from Policy500 of the CUB scan tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using ScanPolicyT =
      AgentScanPolicy<128, 12, AccumT, BLOCK_LOAD_DIRECT, LOAD_CA, BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, BLOCK_SCAN_RAKING>;
  };
};

C2H_TEST("DispatchScan::Dispatch: custom policy hub", "[scan][device]")
{
  using value_t            = int;
  using offset_t           = unsigned;
  using scan_op_t          = cuda::std::plus<>;
  using accum_t            = cuda::std::__accumulator_t<scan_op_t, value_t, value_t>;
  const offset_t num_items = 12345;

  c2h::device_vector<value_t> in_items(num_items);
  c2h::device_vector<value_t> out_items(num_items);
  c2h::gen(C2H_SEED(1), in_items);

  c2h::host_vector<value_t> expected(num_items);
  c2h::host_vector<value_t> host_items(in_items);
  compute_inclusive_scan_reference(host_items.cbegin(), host_items.cend(), expected.begin(), scan_op_t{}, value_t{});

  using policy_hub_t = my_policy_hub<value_t, value_t, accum_t, offset_t, scan_op_t>;
  using dispatch_t =
    DispatchScan<value_t*, value_t*, scan_op_t, NullType, offset_t, accum_t, ForceInclusive::No, policy_hub_t>;
  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(in_items.data()),
    thrust::raw_pointer_cast(out_items.data()),
    scan_op_t{},
    NullType{},
    num_items,
    /* stream */ nullptr);
  c2h::device_vector<uint8_t> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(in_items.data()),
    thrust::raw_pointer_cast(out_items.data()),
    scan_op_t{},
    NullType{},
    num_items,
    /* stream */ nullptr);

  c2h::host_vector<value_t> host_out(out_items);
  REQUIRE(host_out == expected);
}
