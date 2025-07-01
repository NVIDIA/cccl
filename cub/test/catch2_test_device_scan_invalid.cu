// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/iterator/zip_iterator.h>

#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/limits>

#include <cstdint>

#include "c2h/vector.h"
#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include "thrust/detail/raw_pointer_cast.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_inclusive_scan_with_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

using error_count_t = int64_t;

// Element type for scans
template <typename OffsetT>
struct segment
{
  // Make sure that default constructed segments can not be merged
  OffsetT begin = cuda::std::numeric_limits<OffsetT>::min();
  OffsetT end   = cuda::std::numeric_limits<OffsetT>::max();

  // Needed for final comparison with reference
  friend bool operator==(segment left, segment right)
  {
    return left.begin == right.begin && left.end == right.end;
  }

  friend std::ostream& operator<<(std::ostream& os, const segment& seg)
  {
    return os << "[ " << seg.begin << ", " << seg.end << " )";
  }
};

// Needed for data input using fancy iterators
template <typename OffsetT>
struct tuple_to_segment_op
{
  using seg_t = segment<OffsetT>;
  __host__ __device__ seg_t operator()(cuda::std::tuple<OffsetT, OffsetT> interval)
  {
    const auto [begin, end] = interval;
    return {begin, end};
  }
};

// Actual scan operator doing the core test when run on device
template <typename OffsetT>
struct merge_segments_op
{
  using seg_t = segment<OffsetT>;
  __host__ merge_segments_op(error_count_t* error_count)
      : error_count_{*error_count}
  {}
  __host__ __device__ seg_t operator()(seg_t left, seg_t right)
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (if (left.end != right.begin) { error_count_.fetch_add(1, cuda::std::memory_order_relaxed); }));
    return {left.begin, right.end};
  }

  cuda::atomic_ref<error_count_t, cuda::thread_scope::thread_scope_device> error_count_;
};

// Expected to fail for the current implementation.
C2H_TEST("Device scan avoids invalid data with all device interfaces", "[scan][device][!mayfail]")
{
  using offset_t = int32_t;
  using input_t  = segment<offset_t>;
  using output_t = input_t;
  using op_t     = merge_segments_op<offset_t>;

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE_COPY(
    take(3, random(1, 10'000'000)), values({1, 31, cuda::ipow(31, 2), cuda::ipow(31, 4), cuda::ipow(31, 5)}));

  const auto d_in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<offset_t>{1}, cuda::counting_iterator<offset_t>{2}),
    tuple_to_segment_op<offset_t>{});

  SECTION("inclusive scan")
  {
    c2h::device_vector<error_count_t> error_count(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_count.data())};

    // Prepare verification data
    // Need neutral init in this case
    const auto init_value = output_t{1, 1};
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), scan_op, init_value);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan(d_in_it, d_out_it, scan_op, num_items);
    // The actual core requirement currently expected to fail
    REQUIRE(error_count.front() == 0);

    // This one should pass
    REQUIRE(expected_result == out_result);
  }

  SECTION("inclusive scan with init value")
  {
    c2h::device_vector<error_count_t> error_count(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_count.data())};

    const auto init_value = output_t{0, 1};

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), scan_op, init_value);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan_with_init(d_in_it, d_out_it, scan_op, init_value, num_items);
    // The actual core requirement currently expected to fail
    REQUIRE(error_count.front() == 0);

    // This one should pass
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan")
  {
    c2h::device_vector<error_count_t> error_count(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_count.data())};

    const auto init_value = output_t{0, 1};

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), init_value, scan_op);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_scan(d_in_it, d_out_it, scan_op, init_value, num_items);
    // The actual core requirement currently expected to fail
    REQUIRE(error_count.front() == 0);

    // This one should pass
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan with future-init value")
  {
    c2h::device_vector<error_count_t> error_count(1);
    // Scan operator
    auto scan_op = op_t{thrust::raw_pointer_cast(error_count.data())};

    const auto init_value = output_t{0, 1};

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
    // The actual core requirement currently expected to fail
    REQUIRE(error_count.front() == 0);

    // This one should pass
    REQUIRE(expected_result == out_result);
  }
}
