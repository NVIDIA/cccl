// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_three_way_partition.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the three way partition dispatcher after publishing the
// tuning API

template <class InputT, class OffsetT>
struct my_policy_hub
{
  // from Policy500 of the CUB three-way partition tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using ThreeWayPartitionPolicy =
      AgentThreeWayPartitionPolicy<256,
                                   Nominal4BItemsToItems<InputT>(9),
                                   BLOCK_LOAD_DIRECT,
                                   LOAD_DEFAULT,
                                   BLOCK_SCAN_WARP_SCANS,
                                   cub::detail::fixed_delay_constructor_t<350, 450>>;
  };
};

struct less_than_zero_t
{
  __host__ __device__ bool operator()(int value) const
  {
    return value < 0;
  }
};

struct equal_zero_t
{
  __host__ __device__ bool operator()(int value) const
  {
    return value == 0;
  }
};

C2H_TEST("DispatchThreeWayPartitionIf::Dispatch: custom policy hub", "[partition][device]")
{
  using value_t  = int;
  using offset_t = int;

  const c2h::host_vector<value_t> h_in{3, -1, 0, 2, -2, 5, 0, 4};
  c2h::device_vector<value_t> d_in = h_in;
  c2h::device_vector<value_t> d_first(h_in.size());
  c2h::device_vector<value_t> d_second(h_in.size());
  c2h::device_vector<value_t> d_unselected(h_in.size());
  c2h::device_vector<offset_t> d_num_selected(2, 0);

  c2h::host_vector<value_t> expected_first;
  c2h::host_vector<value_t> expected_second;
  c2h::host_vector<value_t> expected_unselected;
  for (const auto value : h_in)
  {
    if (less_than_zero_t{}(value))
    {
      expected_first.push_back(value);
    }
    else if (equal_zero_t{}(value))
    {
      expected_second.push_back(value);
    }
    else
    {
      expected_unselected.push_back(value);
    }
  }

  using policy_hub_t = my_policy_hub<value_t, offset_t>;
  using dispatch_t   = DispatchThreeWayPartitionIf<
      value_t*,
      value_t*,
      value_t*,
      value_t*,
      offset_t*,
      less_than_zero_t,
      equal_zero_t,
      offset_t,
      policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_first.data()),
    thrust::raw_pointer_cast(d_second.data()),
    thrust::raw_pointer_cast(d_unselected.data()),
    thrust::raw_pointer_cast(d_num_selected.data()),
    less_than_zero_t{},
    equal_zero_t{},
    static_cast<offset_t>(h_in.size()),
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_first.data()),
    thrust::raw_pointer_cast(d_second.data()),
    thrust::raw_pointer_cast(d_unselected.data()),
    thrust::raw_pointer_cast(d_num_selected.data()),
    less_than_zero_t{},
    equal_zero_t{},
    static_cast<offset_t>(h_in.size()),
    /* stream */ nullptr);

  const auto num_first  = static_cast<std::size_t>(d_num_selected[0]);
  const auto num_second = static_cast<std::size_t>(d_num_selected[1]);

  d_first.resize(num_first);
  d_second.resize(num_second);
  d_unselected.resize(h_in.size() - num_first - num_second);

  REQUIRE(d_first == expected_first);
  REQUIRE(d_second == expected_second);
  REQUIRE(d_unselected == expected_unselected);
}
