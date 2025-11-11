// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file tests calling cub::DispatchReduce directly

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_reduce.cuh>

#include <cstdint>

#include "catch2_test_device_reduce.cuh"
#include <c2h/catch2_test_helper.h>

using value_types = c2h::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t, float, double>;

template <typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    using ReducePolicy =
      cub::AgentReducePolicy<threads_per_block,
                             items_per_thread,
                             AccumT,
                             items_per_vec_load,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             cub::LOAD_DEFAULT>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = policy_t;
};

C2H_TEST("Dispatch reduce can be called with custom policy_hub", "[reduce][device]", value_types)
{
  using T        = c2h::get<0, TestType>;
  using offset_t = int32_t;
  using init_t   = T;
  using op_t     = cuda::std::plus<>;
  using accum_t  = cuda::std::__accumulator_t<op_t, T, T>;

  const int num_items = 12'345;

  // Prepare input data and output
  c2h::device_vector<T> in_items(num_items, 42);
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  c2h::device_vector<T> out_result(1);
  auto d_out_it = unwrap_it(thrust::raw_pointer_cast(out_result.data()));

  // Run test
  using dispatch_t =
    cub::DispatchReduce<decltype(d_in_it),
                        decltype(d_out_it),
                        offset_t,
                        op_t,
                        init_t,
                        accum_t,
                        ::cuda::std::identity,
                        policy_hub_t<accum_t>>;

  size_t temp_storage_bytes = 0;
  dispatch_t::Dispatch(nullptr, temp_storage_bytes, d_in_it, d_out_it, num_items, op_t{}, init_t{}, 0);

  c2h::device_vector<char> temp_storage(temp_storage_bytes, thrust::no_init);
  dispatch_t::Dispatch(temp_storage.data().get(), temp_storage_bytes, d_in_it, d_out_it, num_items, op_t{}, init_t{}, 0);

  // Verify result
  const T expected_result = static_cast<T>(compute_single_problem_reference(in_items, op_t{}, accum_t{}));
  REQUIRE(expected_result == out_result[0]);
}
