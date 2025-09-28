// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::Reduce, device_reduce);

// huge_data<x> -> sizeof(AgentReduce<...>::TempStorage) on sm86
// 16 -> 224
// 32 -> 368
// 64 -> 656
// 4Ki -> 8216
// 16Ki -> 32792
// 24Ki -> 49176
// 32Ki -> 65560 (but causes Error: Formal parameter space overflowed)
using value_types =
  c2h::type_list<c2h::custom_type_t<c2h::equal_comparable_t, c2h::accumulateable_t, c2h::huge_data<24 * 1024>::type>>;

C2H_TEST("DeviceReduce::Reduce works for large types", "[reduce][device]", value_types)
{
  using value_t  = c2h::get<0, TestType>;
  using offset_t = int;
  using op_t     = cuda::std::plus<>;

  // ensure we hit the vsmem code path
  using policy_hub_t = cub::detail::reduce::policy_hub<value_t, offset_t, op_t>;
  REQUIRE(cub::detail::invoke_for_each_active_policy<policy_hub_t>([]([[maybe_unused]] auto active_policy) {
            using agent_reduce_t = cub::detail::reduce::
              AgentReduce<typename decltype(active_policy)::SingleTilePolicy, value_t*, offset_t, op_t, value_t>;
            using vsmem_helper_t = cub::detail::vsmem_helper_impl<agent_reduce_t>;
            STATIC_REQUIRE(vsmem_helper_t::needs_vsmem);
          })
          == cudaSuccess);

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(take(2, random(1, 1000)), values({1, 32, 64, 128, 256, 512}));
  c2h::device_vector<value_t> values(num_items, thrust::default_init);
  c2h::gen(C2H_SEED(2), values);

  // Prepare verification data
  c2h::host_vector<value_t> h_values(values);
  const auto expected_result = std::reduce(h_values.begin(), h_values.end(), value_t{}, op_t{});

  // Run test
  c2h::device_vector<value_t> out_result(1, thrust::default_init);
  device_reduce(
    thrust::raw_pointer_cast(values.data()), thrust::raw_pointer_cast(out_result.data()), num_items, op_t{}, value_t{});

  // Verify result
  REQUIRE(expected_result == out_result[0]);
}
