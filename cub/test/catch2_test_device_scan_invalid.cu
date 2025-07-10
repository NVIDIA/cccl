// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/limits>

#include <cstdint>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/vector.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_inclusive_scan_with_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

// Element type for scans
struct segment;

// atomicAdd does not support uint64_t or int64_t or long long
using error_count_t    = unsigned long long;
using segment_offset_t = int32_t;
using primitive_t      = uint64_t;
using element_types    = c2h::type_list<primitive_t, segment>;

static_assert(!cub::detail::is_primitive_v<segment>);
static_assert(cub::detail::is_primitive_v<primitive_t>);
static_assert(sizeof(primitive_t) == 2 * sizeof(segment_offset_t));

struct segment
{
  // Make sure that default constructed segments can not be merged
  segment_offset_t begin = cuda::std::numeric_limits<segment_offset_t>::min();
  segment_offset_t end   = cuda::std::numeric_limits<segment_offset_t>::max();

  __host__ __device__ friend bool operator==(segment left, segment right)
  {
    return left.begin == right.begin && left.end == right.end;
  }

  // Needed for final comparison with reference
  friend std::ostream& operator<<(std::ostream& os, const segment& seg)
  {
    return os << "[ " << seg.begin << ", " << seg.end << " )";
  }
};

// cuda::std::bitcast on platforms w/o __builtin_bit_cast needs To to have a trivial default constructor which segment
// does not have by design.
template <typename To, typename From>
__host__ __device__ To dangerous_bit_cast(const From& from)
{
  static_assert(sizeof(From) == sizeof(To));
  To to;
  memcpy(static_cast<void*>(&to), &from, sizeof(To));
  return to;
}
// Needed for data input using fancy iterators
template <typename WrapperT>
struct tuple_to_wrapper_op
{
  __host__ __device__ auto operator()(cuda::std::tuple<segment_offset_t, segment_offset_t> interval)
  {
    const auto [begin, end] = interval;
    return dangerous_bit_cast<WrapperT>(segment{begin, end});
  }
};

struct counts
{
  error_count_t default_init;
  error_count_t zero_init;
  error_count_t other; // Duplicating elements or otherwise combining valid elements in an invalid way/order.
  error_count_t cascade; // Counting down-the-line errors to avoid muddying the other counters
};

// Actual scan operator doing the core test when run on device
struct merge_segments_op
{
  static constexpr auto cascaded = segment{-1, -1};

  __host__ merge_segments_op(counts* error_counts)
      : error_counts_{error_counts}
  {}

  __device__ bool check_inputs(segment left, segment right)
  {
    // Can't avoid left == right check due to potential zero-initialized segments.
    if (left.end != right.begin || left == right)
    {
      error_count_t* error_count_ptr = &error_counts_->other;
      if (left == cascaded || right == cascaded)
      {
        error_count_ptr = &error_counts_->cascade;
      }
      else if (left == segment{} || right == segment{})
      {
        error_count_ptr = &error_counts_->default_init;
      }
      else if (left == segment{0, 0} || right == segment{0, 0})
      {
        error_count_ptr = &error_counts_->zero_init;
      }
      atomicAdd(error_count_ptr, error_count_t{1});

      return true;
    }
    return false;
  }

  __host__ __device__ segment operator()(segment left, segment right)
  {
    bool error_found = false;
    NV_IF_TARGET(NV_IS_DEVICE, (error_found = check_inputs(left, right);));
    if (error_found)
    {
      return cascaded;
    }
    return {left.begin, right.end};
  }
  __host__ __device__ primitive_t operator()(primitive_t p_left, primitive_t p_right)
  {
    const auto left  = dangerous_bit_cast<segment>(p_left);
    const auto right = dangerous_bit_cast<segment>(p_right);
    return dangerous_bit_cast<primitive_t>(this->operator()(left, right));
  }

  counts* error_counts_;
};

// Expected to fail for the current implementation.
C2H_TEST("Device scan avoids invalid data with all device interfaces", "[scan][device][!mayfail]", element_types)
{
  using input_t  = c2h::get<0, TestType>;
  using output_t = input_t;
  using op_t     = merge_segments_op;
  INFO("input_t is " << (cub::detail::is_primitive_v<input_t> ? "primitive" : "not primitive"));

  // Generate the input sizes to test for
  const segment_offset_t num_items = GENERATE_COPY(
    take(3, random(1, 10'000'000)), values({1, 31, cuda::ipow(31, 2), cuda::ipow(31, 4), cuda::ipow(31, 5)}));
  CAPTURE(num_items);

  const auto d_in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment_offset_t>{1},
                              cuda::counting_iterator<segment_offset_t>{2}),
    tuple_to_wrapper_op<input_t>{});

  SECTION("inclusive scan")
  {
    c2h::device_vector<counts> error_counts(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_counts.data())};

    // Prepare verification data
    // Need neutral init in this case
    const auto init_value = dangerous_bit_cast<output_t>(segment{1, 1});
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), scan_op, init_value);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan(d_in_it, d_out_it, scan_op, num_items);

    const counts h_counts = error_counts.front();
    // The actual core requirements currently expected to fail
    CHECK(h_counts.default_init == error_count_t{});
    CHECK(h_counts.zero_init == error_count_t{});
    CHECK(h_counts.other == error_count_t{});
    REQUIRE(h_counts.cascade == error_count_t{});

    // This one should pass
    REQUIRE(expected_result == out_result);
  }

  SECTION("inclusive scan with init value")
  {
    c2h::device_vector<counts> error_counts(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_counts.data())};

    const auto init_value = dangerous_bit_cast<output_t>(segment{0, 1});

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), scan_op, init_value);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan_with_init(d_in_it, d_out_it, scan_op, init_value, num_items);

    const counts h_counts = error_counts.front();
    // The actual core requirements currently expected to fail
    CHECK(h_counts.default_init == error_count_t{});
    CHECK(h_counts.zero_init == error_count_t{});
    CHECK(h_counts.other == error_count_t{});
    REQUIRE(h_counts.cascade == error_count_t{});

    // This one should pass
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan")
  {
    c2h::device_vector<counts> error_counts(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_counts.data())};

    const auto init_value = dangerous_bit_cast<output_t>(segment{0, 1});

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), init_value, scan_op);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_scan(d_in_it, d_out_it, scan_op, init_value, num_items);

    const counts h_counts = error_counts.front();
    // The actual core requirements currently expected to fail
    CHECK(h_counts.default_init == error_count_t{});
    CHECK(h_counts.zero_init == error_count_t{});
    CHECK(h_counts.other == error_count_t{});
    REQUIRE(h_counts.cascade == error_count_t{});

    // This one should pass
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan with future-init value")
  {
    c2h::device_vector<counts> error_counts(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_counts.data())};

    const auto init_value = dangerous_bit_cast<output_t>(segment{0, 1});

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), init_value, scan_op);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    using init_t        = output_t;
    c2h::device_vector<init_t> d_initial_value{init_value};
    const auto future_init_value = cub::FutureValue<init_t>(thrust::raw_pointer_cast(d_initial_value.data()));
    device_exclusive_scan(d_in_it, d_out_it, scan_op, future_init_value, num_items);

    const counts h_counts = error_counts.front();
    // The actual core requirements currently expected to fail
    CHECK(h_counts.default_init == error_count_t{});
    CHECK(h_counts.zero_init == error_count_t{});
    CHECK(h_counts.other == error_count_t{});
    REQUIRE(h_counts.cascade == error_count_t{});

    // This one should pass
    REQUIRE(expected_result == out_result);
  }
}
