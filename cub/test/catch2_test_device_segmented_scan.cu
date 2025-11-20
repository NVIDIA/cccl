// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <cstdint>
#include <iostream>
#include <typeinfo>
#include <utility>

#include "catch2_test_device_reduce.cuh" // for reference_extended_fp
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/extended_types.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_inclusive_segmented_sum);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedSum, device_exclusive_segmented_sum);

DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScan, device_inclusive_segmented_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::ExclusiveSegmentedScan, device_exclusive_segmented_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceSegmentedScan::InclusiveSegmentedScanInit, device_inclusive_segmented_scan_with_init);

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

template <typename ValueT, typename OffsetT>
bool check_segment(const c2h::host_vector<ValueT>& h_output,
                   const c2h::host_vector<ValueT>& h_ref,
                   OffsetT begin_offset,
                   OffsetT end_offset)
{
  bool correct = true;
  for (OffsetT pos = begin_offset; pos < end_offset; ++pos)
  {
    if constexpr (cuda::std::is_floating_point_v<ValueT>)
    {
      ValueT ref_v  = h_ref[pos];
      ValueT act_v  = h_output[pos];
      ValueT diff   = (ref_v - act_v);
      ValueT adiff  = (diff > ValueT{0}) ? diff : -diff;
      ValueT ref_av = (ref_v > ValueT{0}) ? ref_v : -ref_v;
      ValueT act_av = (act_v > ValueT{0}) ? act_v : -act_v;

      ValueT eps = ::cuda::std::numeric_limits<ValueT>::epsilon();
      correct    = correct && (adiff < 3 * eps + 2 * eps * (::cuda::std::max(ref_av, act_av)));
    }
    else if constexpr (cuda::std::is_same_v<ValueT, half_t> || cuda::std::is_same_v<ValueT, bfloat16_t>)
    {
      float ref_v = h_ref[pos];
      float act_v = h_output[pos];
      if (cuda::std::isfinite(ref_v) && cuda::std::isfinite(act_v))
      {
        float diff   = (ref_v - act_v);
        float adiff  = (diff > float{0}) ? diff : -diff;
        float ref_av = (ref_v > float{0}) ? ref_v : -ref_v;
        float act_av = (act_v > float{0}) ? act_v : -act_v;

        float eps = float{1} / float{128};
        correct   = correct && (adiff < 3 * eps + 5 * eps * (::cuda::std::max(ref_av, act_av)));
      }
    }
    else
    {
      correct = correct && (h_ref[pos] == h_output[pos]);
    }
    if (!correct)
    {
      break;
    }
  }
  return correct;
}

using offsets = c2h::type_list<std::int32_t, std::uint64_t>;

C2H_TEST("Device segmented_scan works with all device interfaces", "[segmented][scan][device]", full_type_list, offsets)
{
  using params   = params_t<TestType>;
  using input_t  = typename params::item_t;
  using output_t = typename params::output_t;
  using offset_t = typename c2h::get<1, TestType>;

  constexpr offset_t min_items = 2 * 1024;
  constexpr offset_t max_items = 384 * 1024;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const offset_t small_size  = num_items / 1024;
  const offset_t medium_size = num_items / 128;
  const offset_t large_size  = num_items / 16;

  assert(small_size > 0);

  // Range of segment sizes to generate
  // Note that the segment range [0, 1] may also include one last segment with more than 1 items
  const std::tuple<offset_t, offset_t> seg_size_range =
    GENERATE_COPY(table<offset_t, offset_t>({{0, small_size}, {medium_size, large_size}, {large_size, num_items}}));
  INFO("Test seg_size_range: [" << std::get<0>(seg_size_range) << ", " << std::get<1>(seg_size_range) << ")");

  // Generate input segments
  c2h::device_vector<offset_t> d_segment_offsets = c2h::gen_uniform_offsets<offset_t>(
    C2H_SEED(1), num_items, std::get<0>(seg_size_range), std::get<1>(seg_size_range));
  const offset_t num_segments = static_cast<offset_t>(d_segment_offsets.size() - 1);
  auto d_offsets_it           = thrust::raw_pointer_cast(d_segment_offsets.data());

  INFO("Num segments: " << num_segments);
  INFO("Types: " << typeid(input_t).name() << " " << typeid(output_t).name() << " " << typeid(offset_t).name());

  // Generate input data
  c2h::device_vector<input_t> in_items(num_items);
  c2h::gen(C2H_SEED(2), in_items);
  auto d_in_it = thrust::raw_pointer_cast(in_items.data());

  c2h::device_vector<output_t> output_vec(num_items);
  auto d_out_it = thrust::raw_pointer_cast(output_vec.data());

  c2h::host_vector<offset_t> h_segment_offsets = d_segment_offsets;
  c2h::host_vector<input_t> h_input            = in_items;
  c2h::host_vector<output_t> h_output(num_items);
  c2h::host_vector<output_t> h_ref(num_items);

  SECTION("exclusive scan")
  {
    using op_t = ::cuda::minimum<>;

    // check 3 offset iterators API
    device_exclusive_segmented_scan(
      d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments, op_t{}, output_t{});

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_exclusive_scan_reference(
        h_input.cbegin() + h_segment_offsets[i],
        h_input.cbegin() + h_segment_offsets[i + 1],
        h_ref.begin() + h_segment_offsets[i],
        output_t{},
        op_t{});

      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_exclusive_segmented_scan(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, num_segments, op_t{}, output_t{});

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }
  }

  SECTION("inclusive scan")
  {
    using op_t    = ::cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // check 3 offset iterators API
    device_inclusive_segmented_scan(
      unwrap_it(d_in_it), unwrap_it(d_out_it), d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments, scan_op);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_inclusive_scan_reference(
        h_input.cbegin() + h_segment_offsets[i],
        h_input.cbegin() + h_segment_offsets[i + 1],
        h_ref.begin() + h_segment_offsets[i],
        scan_op,
        accum_t{});

      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }

    if constexpr (::cuda::std::is_same_v<input_t, output_t>)
    {
      output_vec = in_items;

      // check 2 iterators API for in-place scan
      device_inclusive_segmented_scan(
        unwrap_it(d_out_it), unwrap_it(d_out_it), d_offsets_it, d_offsets_it + 1, num_segments, scan_op);

      h_output = output_vec;

      for (offset_t i = 0; i < num_segments; ++i)
      {
        bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
        REQUIRE(correct);
      }
    }
  }

  SECTION("inclusive segmented scan with init")
  {
    using op_t    = cuda::std::plus<>;
    using accum_t = cuda::std::__accumulator_t<op_t, input_t, input_t>;

    INFO("Accum type: " << typeid(accum_t).name());

    // Scan operator
    auto scan_op = unwrap_op(reference_extended_fp(d_in_it), op_t{});

    // Run test
    accum_t init_value{};
    init_default_constant(init_value);

    // check 3 offset iterators API
    device_inclusive_segmented_scan_with_init(
      unwrap_it(d_in_it),
      unwrap_it(d_out_it),
      d_offsets_it,
      d_offsets_it + 1,
      d_offsets_it,
      num_segments,
      scan_op,
      init_value);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_inclusive_scan_reference(
        h_input.cbegin() + h_segment_offsets[i],
        h_input.cbegin() + h_segment_offsets[i + 1],
        h_ref.begin() + h_segment_offsets[i],
        scan_op,
        init_value);
      // Verify result
      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_inclusive_segmented_scan_with_init(
      unwrap_it(d_in_it), unwrap_it(d_out_it), d_offsets_it, d_offsets_it + 1, num_segments, scan_op, init_value);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      // Verify result
      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }
  }

#if ((TEST_TYPES == 0) || (TEST_TYPES == 1))
  SECTION("exclusive sum")
  {
    using op_t = cuda::std::plus<>;

    // check 3 offset iterators API
    device_exclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_exclusive_scan_reference(
        h_input.cbegin() + h_segment_offsets[i],
        h_input.cbegin() + h_segment_offsets[i + 1],
        h_ref.begin() + h_segment_offsets[i],
        output_t{},
        cuda::std::plus<>{});

      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_exclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, num_segments);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }
  }

  SECTION("inclusive sum")
  {
    using op_t = cuda::std::plus<>;

    // check 3 offset iterators API
    device_inclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, d_offsets_it, num_segments);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      compute_inclusive_scan_reference(
        h_input.cbegin() + h_segment_offsets[i],
        h_input.cbegin() + h_segment_offsets[i + 1],
        h_ref.begin() + h_segment_offsets[i],
        cuda::std::plus<>{},
        output_t{});

      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }

    // check 2 offset iterators API
    device_inclusive_segmented_sum(d_in_it, d_out_it, d_offsets_it, d_offsets_it + 1, num_segments);

    h_output = output_vec;

    for (offset_t i = 0; i < num_segments; ++i)
    {
      bool correct = check_segment(h_output, h_ref, h_segment_offsets[i], h_segment_offsets[i + 1]);
      REQUIRE(correct);
    }
  }
#endif
}
