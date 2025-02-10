// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/thread/thread_operators.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "catch2_large_problem_helper.cuh"
#include "catch2_segmented_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_sum);

// %PARAM% TEST_LAUNCH lid 0:1:2

struct get_gaussian_sum_from_offset_op
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ::cuda::std::int64_t
  operator()(::cuda::std::int64_t begin, ::cuda::std::int64_t end)
  {
    // The sum of the first n integers is n(n+1)/2
    ::cuda::std::int64_t length                 = end - begin;
    const ::cuda::std::int64_t section_gaussian = ((begin - 1) * length + length * (length + 1) / 2);
    return section_gaussian;
  }
};

struct custom_sum_op
{
  template <typename ItemT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ItemT operator()(const ItemT lhs, const ItemT rhs) const
  {
    return lhs + rhs;
  }
};

C2H_TEST("Device reduce works with a very large number of segments", "[reduce][device]")
{
  using offset_t        = ::cuda::std::int64_t;
  using segment_index_t = ::cuda::std::int64_t;

  CAPTURE(c2h::type_name<offset_t>());

  constexpr auto num_empty_segments =
    static_cast<segment_index_t>(::cuda::std::numeric_limits<std::uint32_t>::max() - 1000000);
  const auto num_segments = static_cast<segment_index_t>(::cuda::std::numeric_limits<std::uint32_t>::max()) + 2000000;
  constexpr offset_t segment_size = 1000;
  const offset_t num_items        = static_cast<offset_t>(num_segments - num_empty_segments) * segment_size;
  CAPTURE(c2h::type_name<offset_t>(), c2h::type_name<segment_index_t>(), num_items, num_segments, num_empty_segments);

  // Input data
  const auto segment_index_it = thrust::make_counting_iterator(segment_index_t{});

  // Segment offsets
  segment_index_to_offset_op<offset_t, segment_index_t> index_to_offset_op{
    num_empty_segments, num_segments, segment_size, num_items};
  auto offsets_it = thrust::make_transform_iterator(segment_index_it, index_to_offset_op);

  SECTION("segmented reduce")
  {
    using sum_t = ::cuda::std::int64_t;

    // Use a custom operator to increase test coverage
    using op_t = custom_sum_op;

    // Initial value of reduction
    const auto init_val = sum_t{0};

    // Binary reduction operator
    const auto reduction_op = op_t{};

    // Prepare helper to check results
    auto get_sum_from_offset_pair_op = thrust::make_zip_function(get_gaussian_sum_from_offset_op{});
    auto offset_pair_it              = thrust::make_zip_iterator(thrust::make_tuple(offsets_it, offsets_it + 1));
    auto expected_result_it          = thrust::make_transform_iterator(offset_pair_it, get_sum_from_offset_pair_op);
    auto check_result_helper         = detail::large_problem_test_helper(num_segments);
    auto check_result_it             = check_result_helper.get_flagging_output_iterator(expected_result_it);

    // Run test
    const auto input_it = thrust::make_counting_iterator(sum_t{});
    device_segmented_reduce(input_it, check_result_it, num_segments, offsets_it, offsets_it + 1, reduction_op, init_val);

    // Verify all results were written as expected
    check_result_helper.check_all_results_correct();
  }

  SECTION("segmented sum")
  {
    using sum_t = ::cuda::std::int64_t;

    // Prepare helper to check results
    auto get_sum_from_offset_pair_op = thrust::make_zip_function(get_gaussian_sum_from_offset_op{});
    auto offset_pair_it              = thrust::make_zip_iterator(thrust::make_tuple(offsets_it, offsets_it + 1));
    auto expected_result_it          = thrust::make_transform_iterator(offset_pair_it, get_sum_from_offset_pair_op);
    auto check_result_helper         = detail::large_problem_test_helper(num_segments);
    auto check_result_it             = check_result_helper.get_flagging_output_iterator(expected_result_it);

    // Run test
    const auto input_it = thrust::make_counting_iterator(sum_t{});
    device_segmented_sum(input_it, check_result_it, num_segments, offsets_it, offsets_it + 1);

    // Verify all results were written as expected
    check_result_helper.check_all_results_correct();
  }
}
