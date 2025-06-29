
/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuda/cmath>
#include <cuda/std/limits>

#include <cstdint>

#include "catch2_test_device_scan.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScanInit, device_inclusive_scan_with_init);

// %PARAM% TEST_LAUNCH lid 0:1:2

// Element type for scans
template <typename OffsetT>
struct segment
{
  // Make sure that default constructed segments can not be merged
  OffsetT begin = cuda::std::numeric_limits<OffsetT>::min();
  OffsetT end   = cuda::std::numeric_limits<OffsetT>::max();
};

// Needed for final comparison with reference
template <typename OffsetT>
bool operator==(Segment<OffsetT> left, Segment<OffsetT> right)
{
  return left.begin == right.begin && left.end == right.end;
}
template <typename OffsetT>
std::ostream& operator<<(std::ostream& os, const Segment<OffsetT>& seg)
{
  return os << "[ " << seg.begin << ", " << seg.end << " )";
}

// Needed for data input using fancy iterators
template <typename OffsetT>
struct Tuple2Seg
{
  using seg_t = Segment<OffsetT>;
  __host__ __device__ seg_t operator()(thrust::tuple<OffsetT, OffsetT> interval)
  {
    return {.begin = thrust::get<0>(interval), .end = thrust::get<1>(interval)};
  }
};

// Actual scan operator doing the core test when run on device
template <typename OffsetT>
struct SegMerge
{
  using seg_t = Segment<OffsetT>;
  __host__ __device__ seg_t operator()(seg_t left, seg_t right)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (if (left.end != right.begin) { __trap(); }));
    return {.begin = left.begin, .end = right.end};
  }
};

// Expected to fail for the current implementation.
C2H_TEST("Device scan avoids invalid data with all device interfaces", "[scan][device][!mayfail]")
{
  using offset_t = int32_t;
  using input_t  = Segment<offset_t>;
  using output_t = input_t;
  using op_t     = SegMerge<offset_t>;

  // Scan operator
  auto scan_op = op_t{};

  // Generate the input sizes to test for
  const offset_t num_items = GENERATE(1, 31, cuda::ipow(31, 2), cuda::ipow(31, 4), cuda::ipow(31, 5));

  const auto d_in_it = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::counting_iterator<offset_t>{1}, thrust::counting_iterator<offset_t>{2}),
    Tuple2Seg<offset_t>{});

  SECTION("inclusive scan")
  {
    // Prepare verification data
    // Need neutral init in this case
    auto const init_value = output_t{.begin = 1, .end = 1};
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), scan_op, init_value);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan(d_in_it, d_out_it, scan_op, num_items);

    // The actual core requirement is implicitly checked inside the launch wrapper due to __trap().
    // This one would pass already if __trap() would not abort the scan-kernel as well as the test.
    REQUIRE(expected_result == out_result);
  }

  SECTION("inclusive scan with init value")
  {
    auto const init_value = output_t{.begin = 0, .end = 1};

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_inclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), scan_op, init_value);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_inclusive_scan_with_init(d_in_it, d_out_it, scan_op, init_value, num_items);

    // The actual core requirement is implicitly checked inside the launch wrapper due to __trap().
    // This one would pass already if __trap() would not abort the scan-kernel as well as the test.
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan")
  {
    auto const init_value = output_t{.begin = 0, .end = 1};

    // Prepare verification data
    c2h::host_vector<output_t> expected_result(num_items);
    compute_exclusive_scan_reference(d_in_it, d_in_it + num_items, expected_result.begin(), init_value, scan_op);

    // Run test
    c2h::device_vector<output_t> out_result(num_items);
    const auto d_out_it = thrust::raw_pointer_cast(out_result.data());
    device_exclusive_scan(d_in_it, d_out_it, scan_op, init_value, num_items);

    // The actual core requirement is implicitly checked inside the launch wrapper due to __trap().
    // This one would pass already if __trap() would not abort the scan-kernel as well as the test.
    REQUIRE(expected_result == out_result);
  }

  SECTION("exclusive scan with future-init value")
  {
    auto const init_value = output_t{.begin = 0, .end = 1};

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

    // The actual core requirement is implicitly checked inside the launch wrapper due to __trap().
    // This one would pass already if __trap() would not abort the scan-kernel as well as the test.
    REQUIRE(expected_result == out_result);
  }
}
