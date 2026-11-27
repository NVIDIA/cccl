// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include "catch2_large_problem_helper.cuh"
#include "cub_test_macros.h"

// A policy with 96 threads per block, so that the number of segments per block (96 for small, 3 for medium segments) is
// not a power of two. With INT32_MAX segments in a single kernel invocation, the last thread block would then compute
// segment ids beyond INT32_MAX, which overflow if the dispatch does not limit the segments per invocation to a multiple
// of the segments per block.
struct non_power_of_two_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> cub::SegmentedReducePolicy
  {
    constexpr int threads_per_block = 96;
    return cub::SegmentedReducePolicy{
      cub::ReducePassPolicy{threads_per_block, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT},
      cub::SegmentedReduceWarpReducePolicy{threads_per_block, 32, 1, 1, cub::LOAD_DEFAULT},
      cub::SegmentedReduceWarpReducePolicy{threads_per_block, 1, 1, 1, cub::LOAD_DEFAULT}};
  }
};

CUB_TEST("DeviceSegmentedReduce handles INT32_MAX segments with a non-power-of-two number of segments per block",
         "[segmented][reduce][device]",
         CUB_SMALL)
{
  using offset_t = int;
  using value_t  = ::cuda::std::int64_t;
  using op_t     = ::cuda::std::plus<>;

  const auto num_segments = static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<offset_t>::max());

  // Segment i consists of the single element i, so the expected result of segment i is i
  const auto in_it = cuda::counting_iterator{value_t{0}};
  detail::large_problem_test_helper helper(static_cast<std::size_t>(num_segments));
  auto out_it = helper.get_flagging_output_iterator(cuda::counting_iterator{value_t{0}});

  SECTION("fixed size segments")
  {
    constexpr offset_t segment_size = 1;

    std::size_t temp_size = 0;
    REQUIRE(
      cudaSuccess
      == cub::detail::segmented_reduce::dispatch_fixed_size<value_t>(
        nullptr,
        temp_size,
        in_it,
        out_it,
        num_segments,
        segment_size,
        op_t{},
        value_t{0},
        nullptr,
        non_power_of_two_policy_selector{}));
    c2h::device_vector<::cuda::std::uint8_t> temp(temp_size, thrust::no_init);
    REQUIRE(
      cudaSuccess
      == cub::detail::segmented_reduce::dispatch_fixed_size<value_t>(
        thrust::raw_pointer_cast(temp.data()),
        temp_size,
        in_it,
        out_it,
        num_segments,
        segment_size,
        op_t{},
        value_t{0},
        nullptr,
        non_power_of_two_policy_selector{}));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    helper.check_all_results_correct();
  }

  SECTION("variable size segments with a max segment size hint")
  {
    const auto begin_offsets               = cuda::counting_iterator{offset_t{0}};
    const auto end_offsets                 = cuda::counting_iterator{offset_t{1}};
    constexpr std::size_t max_segment_size = 1;

    std::size_t temp_size = 0;
    REQUIRE(
      cudaSuccess
      == cub::detail::segmented_reduce::dispatch<value_t>(
        nullptr,
        temp_size,
        in_it,
        out_it,
        num_segments,
        begin_offsets,
        end_offsets,
        op_t{},
        value_t{0},
        max_segment_size,
        nullptr,
        non_power_of_two_policy_selector{}));
    c2h::device_vector<::cuda::std::uint8_t> temp(temp_size, thrust::no_init);
    REQUIRE(
      cudaSuccess
      == cub::detail::segmented_reduce::dispatch<value_t>(
        thrust::raw_pointer_cast(temp.data()),
        temp_size,
        in_it,
        out_it,
        num_segments,
        begin_offsets,
        end_offsets,
        op_t{},
        value_t{0},
        max_segment_size,
        nullptr,
        non_power_of_two_policy_selector{}));
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    helper.check_all_results_correct();
  }
}
