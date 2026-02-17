// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/functional>

#include "catch2_test_device_reduce.cuh"
#include <c2h/catch2_test_helper.h>

using namespace cub;

// TODO(bgruber): drop this test with CCCL 4.0 when we drop the segmented reduce dispatcher after publishing the
// tuning API

template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct my_policy_hub
{
  // from Policy500 of the CUB segmented reduce tunings
  struct MaxPolicy : ChainedPolicy<500, MaxPolicy, MaxPolicy>
  {
    using ReducePolicy          = AgentReducePolicy<256, 20, AccumT, 4, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG>;
    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };
};

C2H_TEST("DispatchSegmentedReduce::Dispatch: custom policy hub", "[segmented][reduce][device]")
{
  using input_t     = int;
  using output_t    = int;
  using offset_t    = int;
  using reduction_t = cuda::std::plus<>;
  using accum_t     = cuda::std::__accumulator_t<reduction_t, input_t, output_t>;

  c2h::device_vector<offset_t> offsets{0, 3, 3, 7, 9, 15};
  c2h::device_vector<input_t> in_items{
    8, 6, 7, 5, 3, 0, 9, 25, 24, 6, 7, 2, 46, 8, 123, 2, 5, 3, 76, 48,
  };
  const auto num_segments = static_cast<::cuda::std::int64_t>(offsets.size() - 1);

  c2h::device_vector<output_t> out_result(num_segments, thrust::no_init);

  c2h::host_vector<output_t> expected_result(num_segments, thrust::no_init);
  compute_segmented_problem_reference(in_items, offsets, reduction_t{}, accum_t{}, expected_result.begin());

  using policy_hub_t = my_policy_hub<accum_t, offset_t, reduction_t>;
  using dispatch_t   = DispatchSegmentedReduce<
      input_t*,
      output_t*,
      const offset_t*,
      const offset_t*,
      offset_t,
      reduction_t,
      output_t,
      accum_t,
      policy_hub_t>;

  size_t temp_size = 0;
  dispatch_t::Dispatch(
    nullptr,
    temp_size,
    thrust::raw_pointer_cast(in_items.data()),
    thrust::raw_pointer_cast(out_result.data()),
    num_segments,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    reduction_t{},
    output_t{},
    /* stream */ nullptr);
  c2h::device_vector<unsigned char> temp_storage(temp_size, thrust::no_init);
  dispatch_t::Dispatch(
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_size,
    thrust::raw_pointer_cast(in_items.data()),
    thrust::raw_pointer_cast(out_result.data()),
    num_segments,
    thrust::raw_pointer_cast(offsets.data()),
    thrust::raw_pointer_cast(offsets.data()) + 1,
    reduction_t{},
    output_t{},
    /* stream */ nullptr);

  REQUIRE(out_result == expected_result);
}
