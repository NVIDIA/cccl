// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <cuda/std/limits>
#include <cuda/std/utility>

#include <numeric>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Reduce, device_segmented_reduce);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Sum, device_segmented_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Min, device_segmented_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMin, device_segmented_arg_min);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::Max, device_segmented_max);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedReduce::ArgMax, device_segmented_arg_max);

// %PARAM% TEST_LAUNCH lid 0:1:2
// %PARAM% TEST_TYPES types 0:1:2:3

// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t,
                     c2h::equal_comparable_t,
                     c2h::lexicographical_less_comparable_t,
                     c2h::lexicographical_greater_comparable_t>;

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<type_pair<std::uint8_t>, type_pair<std::int8_t, std::int32_t>>;
#elif TEST_TYPES == 1
using full_type_list = c2h::type_list<type_pair<std::int32_t>, type_pair<std::int64_t>>;
#elif TEST_TYPES == 2
using full_type_list =
  c2h::type_list<type_pair<uchar3>,
                 type_pair<
#  if _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                   ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                   >>;
#elif TEST_TYPES == 3
// clang-format off
using full_type_list = c2h::type_list<
type_pair<custom_t>
#if TEST_HALF_T()
, type_pair<half_t> // testing half
#endif // TEST_HALF_T()
#if TEST_BF_T()
, type_pair<bfloat16_t> // testing bf16
#endif // TEST_BF_T()
>;
// clang-format on
#endif

using offsets = c2h::type_list<std::int32_t, std::uint32_t>;

C2H_TEST("Device reduce works with all device interfaces", "[segmented][reduce][device]", full_type_list, offsets)
{
  using type_pair_t = typename c2h::get<0, TestType>;
  using input_t     = typename type_pair_t::input_t;
  using output_t    = typename type_pair_t::output_t;
  using offset_t    = typename c2h::get<1, TestType>;

  constexpr int min_items = 1;
  constexpr int max_items = 1000000;

  // Number of items
  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const int num_items = GENERATE_COPY(
    take(c2h::adjust_seed_count(2), random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  INFO("Test num_items: " << num_items);

  // Range of segment sizes to generate
  // Note that the segment range [0, 1] may also include one last segment with more than 1 items
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{0, 1}, {1, num_items}, {num_items, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << "]");

  // Generate input segments
  c2h::device_vector<offset_t> segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(segment_offsets.size() - 1);
  auto d_offsets_it           = thrust::raw_pointer_cast(segment_offsets.data());

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  c2h::gen(C2H_SEED(2), in_items);
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  SECTION("reduce")
  {
    using op_t = cuda::std::plus<>;

    // Binary reduction operator
    auto reduction_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, output_t>;
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(in_items, segment_offsets, reduction_op, accum_t{}, expected_result.begin());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    using init_t  = cub::detail::it_value_t<decltype(unwrap_it(d_out_it))>;
    device_segmented_reduce(
      unwrap_it(d_in_it), unwrap_it(d_out_it), num_segments, d_offsets_it, d_offsets_it + 1, reduction_op, init_t{});

    // Verify result
    REQUIRE(expected_result == out_result);
  }

// Skip DeviceReduce::Sum tests for extended floating-point types because of unbounded epsilon due
// to pseudo associativity of the addition operation over floating point numbers
#if TEST_TYPES != 3
  SECTION("sum")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, output_t>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(in_items, segment_offsets, op_t{}, accum_t{}, expected_result.begin());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = unwrap_it(thrust::raw_pointer_cast(out_result.data()));
    device_segmented_sum(d_in_it, d_out_it, num_segments, d_offsets_it, d_offsets_it + 1);

    // Verify result
    REQUIRE(expected_result == out_result);
  }
#endif

  SECTION("min")
  {
    using op_t = cuda::minimum<>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(
      in_items, segment_offsets, op_t{}, cuda::std::numeric_limits<input_t>::max(), expected_result.begin());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_segmented_min(unwrap_it(d_in_it), unwrap_it(d_out_it), num_segments, d_offsets_it, d_offsets_it + 1);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("max")
  {
    using op_t = cuda::maximum<>;

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_segments);
    compute_segmented_problem_reference(
      in_items, segment_offsets, op_t{}, cuda::std::numeric_limits<input_t>::lowest(), expected_result.begin());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_segmented_max(unwrap_it(d_in_it), unwrap_it(d_out_it), num_segments, d_offsets_it, d_offsets_it + 1);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("argmax")
  {
    using result_t = cub::KeyValuePair<int, output_t>;

    // Prepare verification data
    c2h::host_vector<result_t> expected_result(num_segments);
    compute_segmented_argmax_reference(in_items, segment_offsets, expected_result.begin());

    // Run test
    c2h::device_vector<result_t> out_result(num_segments);
    device_segmented_arg_max(
      d_in_it, thrust::raw_pointer_cast(out_result.data()), num_segments, d_offsets_it, d_offsets_it + 1);

    // Verify result
    REQUIRE(expected_result == out_result);
  }

  SECTION("argmin")
  {
    using result_t = cub::KeyValuePair<int, output_t>;

    // Prepare verification data
    c2h::host_vector<input_t> host_items(in_items);
    c2h::host_vector<result_t> expected_result(num_segments);
    compute_segmented_argmin_reference(in_items, segment_offsets, expected_result.begin());

    // Run test
    c2h::device_vector<result_t> out_result(num_segments);
    device_segmented_arg_min(
      d_in_it, thrust::raw_pointer_cast(out_result.data()), num_segments, d_offsets_it, d_offsets_it + 1);
    // Verify result
    REQUIRE(expected_result == out_result);
  }
}

C2H_TEST("Device fixed size segmented reduce works with all device interfaces",
         "[segmented][reduce][device]",
         full_type_list)
{
  using type_pair_t    = typename c2h::get<0, TestType>;
  using input_t        = typename type_pair_t::input_t;
  using output_t       = typename type_pair_t::output_t;
  using segment_size_t = int;

  const int max_items = 1 << 22;

  // Use c2h::adjust_seed_count to reduce runtime on sanitizers.
  const segment_size_t segment_size = GENERATE_COPY(
    take(c2h::adjust_seed_count(2), random(1 << 0, 1 << 5)),
    take(c2h::adjust_seed_count(2), random(1 << 5, 1 << 10)),
    take(c2h::adjust_seed_count(2), random(1 << 10, 1 << 15)),
    take(c2h::adjust_seed_count(2), random(1 << 15, 1 << 20)));

  const int num_segments = max_items / segment_size;
  const int num_items    = num_segments * segment_size;

  CAPTURE(num_items, num_segments, segment_size);

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  c2h::gen(C2H_SEED(2), in_items);

  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  SECTION("reduce")
  {
    using op_t = cuda::std::plus<>;

    // Binary reduction operator
    auto reduction_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Prepare verification data
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, output_t>;
    c2h::host_vector<output_t> expected_result(num_segments);
    accum_t default_constant{};
    init_default_constant(default_constant);

    compute_fixed_size_segmented_problem_reference(
      in_items, num_segments, segment_size, reduction_op, default_constant, expected_result.begin());

    // Run test
    c2h::device_vector<output_t> out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(out_result.data());

    using init_t = cub::detail::it_value_t<decltype(unwrap_it(d_out_it))>;
    init_t init  = static_cast<init_t>(*unwrap_it(&default_constant));
    device_segmented_reduce(unwrap_it(d_in_it), unwrap_it(d_out_it), num_segments, segment_size, reduction_op, init);
    // Verify result
    REQUIRE(expected_result == out_result);
  }

// Skip DeviceReduce::Sum tests for extended floating-point types because of unbounded epsilon due
// to pseudo associativity of the addition operation over floating point numbers
#if TEST_TYPES != 3
  SECTION("sum")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, output_t>;

    // Prepare verification data
    c2h::host_vector<output_t> h_expected_result(num_segments);
    compute_fixed_size_segmented_problem_reference(
      in_items, num_segments, segment_size, op_t{}, accum_t{}, h_expected_result.begin());

    // Run test
    c2h::device_vector<output_t> d_out_result(num_segments);
    auto d_out_it = unwrap_it(thrust::raw_pointer_cast(d_out_result.data()));
    device_segmented_sum(d_in_it, d_out_it, num_segments, segment_size);

    c2h::host_vector<output_t> h_out_result(d_out_result);
    // Verify result
    REQUIRE(h_expected_result == h_out_result);
  }
#endif

  SECTION("min")
  {
    using op_t = cuda::minimum<>;

    // Prepare verification data
    c2h::host_vector<output_t> h_expected_result(num_segments);
    compute_fixed_size_segmented_problem_reference(
      in_items,
      num_segments,
      segment_size,
      op_t{},
      cuda::std::numeric_limits<input_t>::max(),
      h_expected_result.begin());

    // Run test
    c2h::device_vector<output_t> d_out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(d_out_result.data());
    device_segmented_min(unwrap_it(d_in_it), unwrap_it(d_out_it), num_segments, segment_size);

    c2h::host_vector<output_t> h_out_result(d_out_result);
    // Verify result
    REQUIRE(h_expected_result == h_out_result);
  }

  SECTION("argmin")
  {
    using result_t = cuda::std::pair<int, output_t>;

    // Prepare verification data
    c2h::host_vector<result_t> h_expected_result(num_segments);
    compute_fixed_size_segmented_argmin_reference(in_items, num_segments, segment_size, h_expected_result.begin());

    // Run test
    c2h::device_vector<result_t> d_out_result(num_segments);
    device_segmented_arg_min(d_in_it, thrust::raw_pointer_cast(d_out_result.data()), num_segments, segment_size);

    c2h::host_vector<result_t> h_out_result(d_out_result);
    // Verify result
    REQUIRE(h_expected_result == h_out_result);
  }

  SECTION("max")
  {
    using op_t = cuda::maximum<>;

    // Prepare verification data
    c2h::host_vector<output_t> h_expected_result(num_segments);
    compute_fixed_size_segmented_problem_reference(
      in_items,
      num_segments,
      segment_size,
      op_t{},
      cuda::std::numeric_limits<input_t>::lowest(),
      h_expected_result.begin());

    // Run test
    c2h::device_vector<output_t> d_out_result(num_segments);
    auto d_out_it = thrust::raw_pointer_cast(d_out_result.data());
    device_segmented_max(unwrap_it(d_in_it), unwrap_it(d_out_it), num_segments, segment_size);

    c2h::host_vector<output_t> h_out_result(d_out_result);
    // Verify result
    REQUIRE(h_expected_result == h_out_result);
  }

  SECTION("argmax")
  {
    using result_t = cuda::std::pair<int, output_t>;

    // Prepare verification data
    c2h::host_vector<result_t> h_expected_result(num_segments);
    compute_fixed_size_segmented_argmax_reference(in_items, num_segments, segment_size, h_expected_result.begin());

    // Run test
    c2h::device_vector<result_t> d_out_result(num_segments);
    device_segmented_arg_max(d_in_it, thrust::raw_pointer_cast(d_out_result.data()), num_segments, segment_size);

    c2h::host_vector<result_t> h_out_result(d_out_result);
    // Verify result
    REQUIRE(h_expected_result == h_out_result);
  }
}
