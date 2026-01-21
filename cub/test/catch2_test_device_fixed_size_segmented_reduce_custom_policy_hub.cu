// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_fixed_size_segmented_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include "catch2_test_device_reduce.cuh"
#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the fixed-size segmented reduce dispatcher after publishing
// the tuning API

template <typename AccumT>
struct my_policy_hub
{
  // from Policy500 of the CUB fixed-size segmented reduce tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using ReducePolicy = AgentReducePolicy<256, 16, AccumT, 1, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG>;

    using SmallReducePolicy = AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS, 1, 16, AccumT, 1, LOAD_LDG>;

    using MediumReducePolicy = AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS, 32, 16, AccumT, 1, LOAD_LDG>;
  };
};

C2H_TEST("DispatchFixedSizeSegmentedReduce::Dispatch: custom policy hub", "[segmented][reduce][device]")
{
  using input_t     = int;
  using output_t    = int;
  using offset_t    = int;
  using reduction_t = ::cuda::std::plus<>;
  using accum_t     = ::cuda::std::__accumulator_t<reduction_t, input_t, output_t>;

  constexpr offset_t segment_size             = 3;
  constexpr ::cuda::std::int64_t num_segments = 4;
  c2h::host_vector<input_t> h_items{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  c2h::device_vector<input_t> d_items = h_items;
  c2h::device_vector<output_t> out_result(num_segments, thrust::no_init);

  c2h::host_vector<output_t> expected_result(num_segments, thrust::no_init);
  for (::cuda::std::int64_t seg = 0; seg < num_segments; ++seg)
  {
    output_t accum = output_t{};
    for (offset_t i = 0; i < segment_size; ++i)
    {
      accum = reduction_t{}(accum, h_items[seg * segment_size + i]);
    }
    expected_result[seg] = accum;
  }

  using policy_hub_t = my_policy_hub<accum_t>;
  using dispatch_t   = cub::detail::reduce::
    DispatchFixedSizeSegmentedReduce<input_t*, output_t*, offset_t, reduction_t, output_t, accum_t, policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(d_items.data()),
    thrust::raw_pointer_cast(out_result.data()),
    num_segments,
    segment_size,
    reduction_t{},
    output_t{},
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(d_items.data()),
    thrust::raw_pointer_cast(out_result.data()),
    num_segments,
    segment_size,
    reduction_t{},
    output_t{},
    /* stream */ nullptr);

  REQUIRE(out_result == expected_result);
}
