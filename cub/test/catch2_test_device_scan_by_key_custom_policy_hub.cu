// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_scan_by_key.cuh>

#include <cuda/std/functional>

#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the scan-by-key dispatcher after publishing the tuning API

template <typename KeysInputIteratorT, typename AccumT>
struct my_policy_hub
{
  using key_t = cub::detail::it_value_t<KeysInputIteratorT>;

  // from Policy500 of the CUB scan-by-key tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using ScanByKeyPolicyT =
      AgentScanByKeyPolicy<128,
                           6,
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_CA,
                           BLOCK_SCAN_WARP_SCANS,
                           BLOCK_STORE_WARP_TRANSPOSE,
                           cub::detail::default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };
};

C2H_TEST("DispatchScanByKey::Dispatch: custom policy hub", "[scan_by_key][device]")
{
  using key_t     = int;
  using value_t   = int;
  using offset_t  = unsigned int;
  using scan_op_t = cuda::std::plus<>;
  using eq_op_t   = cuda::std::equal_to<>;
  using accum_t   = cuda::std::__accumulator_t<scan_op_t, value_t, value_t>;

  c2h::device_vector<key_t> d_keys{1, 1, 23, 2, 2, 1, 35, 2, 4, 67, 8, 2, 6, 8, 434, 6, 8};
  c2h::device_vector<value_t> d_values{3, 4, 1, 2, 3, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  c2h::device_vector<value_t> d_out(d_values.size());

  c2h::host_vector<value_t> expected(d_values.size());
  compute_inclusive_scan_by_key_reference(d_values, d_keys, expected.begin(), scan_op_t{}, eq_op_t{});

  using policy_hub_t = my_policy_hub<key_t*, accum_t>;
  using dispatch_t =
    DispatchScanByKey<key_t*, value_t*, value_t*, eq_op_t, scan_op_t, NullType, offset_t, accum_t, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    thrust::raw_pointer_cast(d_out.data()),
    eq_op_t{},
    scan_op_t{},
    NullType{},
    static_cast<offset_t>(d_values.size()),
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys.data()),
    thrust::raw_pointer_cast(d_values.data()),
    thrust::raw_pointer_cast(d_out.data()),
    eq_op_t{},
    scan_op_t{},
    NullType{},
    static_cast<offset_t>(d_values.size()),
    /* stream */ nullptr);

  REQUIRE(d_out == expected);
}
