// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_unique_by_key.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/functional>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the unique by key dispatcher after publishing the
// tuning API

template <class KeyT, class ValueT>
struct my_policy_hub
{
  // from Policy500 of the CUB unique-by-key tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using UniqueByKeyPolicyT =
      AgentUniqueByKeyPolicy<128,
                             9,
                             BLOCK_LOAD_WARP_TRANSPOSE,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             cub::detail::default_delay_constructor_t<int>>;
  };
};

C2H_TEST("DispatchUniqueByKey::Dispatch: custom policy hub", "[select_unique_by_key][device]")
{
  using key_t    = int;
  using value_t  = int;
  using offset_t = int;
  using eq_op_t  = cuda::std::equal_to<>;

  const c2h::host_vector<key_t> h_keys_in{1, 1, 2, 2, 2, 3, 1};
  const c2h::host_vector<value_t> h_vals_in{10, 11, 20, 21, 22, 30, 40};
  c2h::device_vector<key_t> d_keys_in   = h_keys_in;
  c2h::device_vector<value_t> d_vals_in = h_vals_in;
  c2h::device_vector<key_t> d_keys_out(h_keys_in.size());
  c2h::device_vector<value_t> d_vals_out(h_vals_in.size());
  c2h::device_vector<offset_t> d_num_selected(1);

  c2h::host_vector<key_t> expected_keys;
  c2h::host_vector<value_t> expected_vals;
  for (std::size_t i = 0; i < h_keys_in.size(); ++i)
  {
    if (i == 0 || h_keys_in[i] != h_keys_in[i - 1])
    {
      expected_keys.push_back(h_keys_in[i]);
      expected_vals.push_back(h_vals_in[i]);
    }
  }

  using policy_hub_t = my_policy_hub<key_t, value_t>;
  using dispatch_t =
    DispatchUniqueByKey<key_t*, value_t*, key_t*, value_t*, offset_t*, eq_op_t, offset_t, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_vals_in.data()),
    thrust::raw_pointer_cast(d_keys_out.data()),
    thrust::raw_pointer_cast(d_vals_out.data()),
    thrust::raw_pointer_cast(d_num_selected.data()),
    eq_op_t{},
    static_cast<offset_t>(h_keys_in.size()),
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_vals_in.data()),
    thrust::raw_pointer_cast(d_keys_out.data()),
    thrust::raw_pointer_cast(d_vals_out.data()),
    thrust::raw_pointer_cast(d_num_selected.data()),
    eq_op_t{},
    static_cast<offset_t>(h_keys_in.size()),
    /* stream */ nullptr);

  const auto num_selected = static_cast<std::size_t>(d_num_selected[0]);
  d_keys_out.resize(num_selected);
  d_vals_out.resize(num_selected);
  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_vals_out == expected_vals);
}
