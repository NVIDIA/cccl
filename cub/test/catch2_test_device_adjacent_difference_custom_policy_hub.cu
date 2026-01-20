// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_adjacent_difference.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/functional>
#include <cuda/std/numeric>

#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the adjacent difference dispatcher after publishing the
// tuning API

template <typename InputIteratorT, bool MayAlias>
struct my_policy_hub
{
  using ValueT = cub::detail::it_value_t<InputIteratorT>;

  // from Policy500 of the CUB adjacent difference tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using AdjacentDifferencePolicy =
      AgentAdjacentDifferencePolicy<128,
                                    Nominal8BItemsToItems<ValueT>(7),
                                    BLOCK_LOAD_WARP_TRANSPOSE,
                                    MayAlias ? LOAD_CA : LOAD_LDG,
                                    BLOCK_STORE_WARP_TRANSPOSE>;
  };
};

C2H_TEST("DispatchAdjacentDifference::Dispatch: custom policy hub", "[device][adjacent_difference]")
{
  using value_t            = int;
  using offset_t           = unsigned;
  using difference_op_t    = cuda::std::minus<>;
  const offset_t num_items = 12345;

  c2h::device_vector<value_t> in_items(num_items);
  c2h::device_vector<value_t> out_items(num_items);
  c2h::gen(C2H_SEED(1), in_items);

  c2h::host_vector<value_t> host_in(in_items);
  c2h::host_vector<value_t> expected(num_items);
  cuda::std::adjacent_difference(host_in.begin(), host_in.end(), expected.begin(), cuda::std::minus<value_t>{});

  using policy_hub_t = my_policy_hub<value_t*, /* may_alias */ false>;
  using dispatch_t =
    DispatchAdjacentDifference<value_t*, value_t*, difference_op_t, offset_t, MayAlias::No, ReadOption::Left, policy_hub_t>;
  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(in_items.data()),
    thrust::raw_pointer_cast(out_items.data()),
    num_items,
    difference_op_t{},
    /* stream */ nullptr);
  c2h::device_vector<std::uint8_t> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(in_items.data()),
    thrust::raw_pointer_cast(out_items.data()),
    num_items,
    difference_op_t{},
    /* stream */ nullptr);

  c2h::host_vector<value_t> host_out(out_items);
  REQUIRE(host_out == expected);
}
