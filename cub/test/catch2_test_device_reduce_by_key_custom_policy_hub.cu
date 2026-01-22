// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>

#include <cuda/std/functional>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the reduce-by-key dispatcher after publishing the tuning API

template <class ReductionOpT, class AccumT, class KeyT>
struct my_policy_hub
{
  // from Policy500 of the CUB reduce-by-key tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<128,
                             6,
                             BLOCK_LOAD_DIRECT,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             cub::detail::default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };
};

C2H_TEST("DispatchReduceByKey::Dispatch: custom policy hub", "[reduce_by_key][device]")
{
  using key_t          = int;
  using value_t        = int;
  using offset_t       = int;
  using equality_op_t  = cuda::std::equal_to<>;
  using reduction_op_t = cuda::std::plus<>;
  using accum_t        = cuda::std::__accumulator_t<reduction_op_t, value_t, value_t>;

  const c2h::host_vector<key_t> h_keys_in{1, 1, 2, 2, 2, 3, 3};
  const c2h::host_vector<value_t> h_values_in{1, 2, 3, 4, 5, 6, 7};

  c2h::device_vector<key_t> d_keys_in     = h_keys_in;
  c2h::device_vector<value_t> d_values_in = h_values_in;
  c2h::device_vector<key_t> d_keys_out(h_keys_in.size());
  c2h::device_vector<value_t> d_aggregates_out(h_values_in.size());
  c2h::device_vector<offset_t> d_num_runs(1);

  c2h::host_vector<key_t> expected_keys{1, 2, 3};
  c2h::host_vector<value_t> expected_values{3, 12, 13};

  using policy_hub_t = my_policy_hub<reduction_op_t, accum_t, key_t>;
  using dispatch_t   = cub::DispatchReduceByKey<
      key_t*,
      key_t*,
      value_t*,
      value_t*,
      offset_t*,
      equality_op_t,
      reduction_op_t,
      offset_t,
      accum_t,
      policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_keys_out.data()),
    thrust::raw_pointer_cast(d_values_in.data()),
    thrust::raw_pointer_cast(d_aggregates_out.data()),
    thrust::raw_pointer_cast(d_num_runs.data()),
    equality_op_t{},
    reduction_op_t{},
    static_cast<offset_t>(h_keys_in.size()),
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_keys_in.data()),
    thrust::raw_pointer_cast(d_keys_out.data()),
    thrust::raw_pointer_cast(d_values_in.data()),
    thrust::raw_pointer_cast(d_aggregates_out.data()),
    thrust::raw_pointer_cast(d_num_runs.data()),
    equality_op_t{},
    reduction_op_t{},
    static_cast<offset_t>(h_keys_in.size()),
    /* stream */ nullptr);

  const auto num_runs = static_cast<std::size_t>(d_num_runs[0]);
  d_keys_out.resize(num_runs);
  d_aggregates_out.resize(num_runs);

  REQUIRE(d_keys_out == expected_keys);
  REQUIRE(d_aggregates_out == expected_values);
}
