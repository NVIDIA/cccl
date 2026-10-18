// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_scan.cuh>

#include <thrust/equal.h>
#include <thrust/fill.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/execution>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include "catch2_test_device_segmented_scan_schedules.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceSegmentedScan::InclusiveSegmentedSum, device_segmented_inclusive_sum);

// %PARAM% TEST_LAUNCH lid 0:1:2

namespace
{
using item_t   = cuda::std::uint8_t;
using offset_t = cuda::std::int32_t;

// Every input item is 1, so the inclusive sum at position k of a segment is k + 1, truncated to item_t.
struct expected_ones_sum
{
  cuda::std::int64_t segment_size;

  __host__ __device__ item_t operator()(cuda::std::int64_t index) const
  {
    return static_cast<item_t>(index % segment_size + 1);
  }
};

// Segments 0, num_segments / 2 and num_segments - 1 hold 1000 items each, back to back; all others are empty.
template <bool End>
struct sparse_offset
{
  cuda::std::int64_t num_segments;

  __host__ __device__ offset_t operator()(cuda::std::int64_t segment) const
  {
    const offset_t extent = End ? 1000 : 0;
    if (segment == 0)
    {
      return extent;
    }
    if (segment == num_segments / 2)
    {
      return 1000 + extent;
    }
    if (segment == num_segments - 1)
    {
      return 2000 + extent;
    }
    return 0;
  }
};

constexpr cuda::std::int64_t total_past_max = cuda::std::int64_t{1} << 31;
constexpr offset_t long_segment_items       = static_cast<offset_t>(total_past_max - 3);

// The first four segments hold 2^31 - 3, 1, 1 and 1 items, all reading from input 0 and writing back to back so that
// their outputs cover [0, 2^31). All other segments are empty.
struct one_block_end_offset
{
  __host__ __device__ offset_t operator()(cuda::std::int64_t segment) const
  {
    return segment == 0 ? long_segment_items : (segment < 4 ? 1 : 0);
  }
};

struct one_block_output_offset
{
  __host__ __device__ offset_t operator()(cuda::std::int64_t segment) const
  {
    return segment == 0 || segment > 3 ? 0 : long_segment_items + static_cast<offset_t>(segment - 1);
  }
};

struct one_block_expected
{
  __host__ __device__ item_t operator()(cuda::std::int64_t index) const
  {
    return index < long_segment_items ? static_cast<item_t>(index + 1) : item_t{1};
  }
};
} // namespace

// The two output segments are disjoint and cover 2^31 items, one past the int32 maximum; the inputs overlap.
CUB_TEST("Device segmented scan with load balancing handles a total one past the offset maximum",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  constexpr offset_t segment_items             = offset_t{1} << 30;
  const c2h::device_vector<offset_t> begins    = {0, 0};
  const c2h::device_vector<offset_t> ends      = {segment_items, segment_items};
  const c2h::device_vector<offset_t> out_begin = {0, segment_items};
  const offset_t* d_begins                     = thrust::raw_pointer_cast(begins.data());
  const offset_t* d_ends                       = thrust::raw_pointer_cast(ends.data());
  const offset_t* d_out_begin                  = thrust::raw_pointer_cast(out_begin.data());
  const auto d_in                              = cuda::constant_iterator<item_t>(item_t{1});
  constexpr cuda::std::int64_t num_segments    = 2;

  c2h::device_vector<item_t> d_out(static_cast<size_t>(total_past_max));
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto run = [&](const auto& schedule_env) {
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, false>(
      schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
    thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
    device_segmented_inclusive_sum(
      d_in,
      d_out_ptr,
      d_begins,
      d_ends,
      d_out_begin,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
    const auto expected = cuda::transform_iterator(
      cuda::counting_iterator(cuda::std::int64_t{0}), expected_ones_sum{cuda::std::int64_t{segment_items}});
    REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
  };

  SECTION("with load balancing")
  {
    run(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  }

  SECTION("with load balancing and num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(total_past_max), cub::segmented_scan_load_balancing});
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

CUB_TEST("Device segmented scan with load balancing works with more segments than the offset type holds",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  constexpr cuda::std::int64_t num_segments = cuda::std::int64_t{cuda::std::numeric_limits<offset_t>::max()} + 1000;
  constexpr cuda::std::int64_t num_items    = 3000;
  const auto segment_it                     = cuda::counting_iterator(cuda::std::int64_t{0});
  const auto d_begins                       = cuda::transform_iterator(segment_it, sparse_offset<false>{num_segments});
  const auto d_ends                         = cuda::transform_iterator(segment_it, sparse_offset<true>{num_segments});
  const auto d_in                           = cuda::constant_iterator<item_t>(item_t{1});

  c2h::device_vector<item_t> d_out(num_items);
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto run = [&](const auto& schedule_env) {
    const size_t bytes = schedule_allocation_size<cub::ForceInclusive::No, true>(
      schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_begins, cuda::std::plus<>{}, cub::NullType{});
    thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
    device_segmented_inclusive_sum(
      d_in,
      d_out_ptr,
      d_begins,
      d_ends,
      num_segments,
      cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});
    const auto expected =
      cuda::transform_iterator(cuda::counting_iterator(cuda::std::int64_t{0}), expected_ones_sum{1000});
    REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
  };

  SECTION("with load balancing")
  {
    run(cuda::std::execution::env{cub::segmented_scan_load_balancing});
  }

  SECTION("with num_items")
  {
    run(cuda::std::execution::env{cub::segmented_scan_num_items(num_items)});
  }
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}

// num_items = 2^31 over 2^26 segments gives a mean of 32 items, so a block takes at least eight consecutive segments
// for any tile of 128 items or more. The first four segments are nonempty, so they share block 0 and sum to 2^31.
CUB_TEST("Device segmented scan with num_items handles a block whose segments sum past the offset maximum",
         "[segmented][scan][device]",
         CUB_LARGE)
try
{
  constexpr cuda::std::int64_t num_segments = cuda::std::int64_t{1} << 26;
  const auto segment_it                     = cuda::counting_iterator(cuda::std::int64_t{0});
  const auto d_begins                       = cuda::constant_iterator<offset_t>(offset_t{0});
  const auto d_ends                         = cuda::transform_iterator(segment_it, one_block_end_offset{});
  const auto d_out_begin                    = cuda::transform_iterator(segment_it, one_block_output_offset{});
  const auto d_in                           = cuda::constant_iterator<item_t>(item_t{1});

  c2h::device_vector<item_t> d_out(static_cast<size_t>(total_past_max));
  item_t* d_out_ptr = thrust::raw_pointer_cast(d_out.data());

  const auto schedule_env = cuda::std::execution::env{cub::segmented_scan_num_items(total_past_max)};
  const size_t bytes      = schedule_allocation_size<cub::ForceInclusive::No, false>(
    schedule_env, d_in, d_out_ptr, num_segments, d_begins, d_ends, d_out_begin, cuda::std::plus<>{}, cub::NullType{});
  thrust::fill(c2h::device_policy, d_out.begin(), d_out.end(), item_t{0});
  device_segmented_inclusive_sum(
    d_in,
    d_out_ptr,
    d_begins,
    d_ends,
    d_out_begin,
    num_segments,
    cuda::std::execution::env{expected_allocation_size(bytes), schedule_env});

  const auto expected = cuda::transform_iterator(cuda::counting_iterator(cuda::std::int64_t{0}), one_block_expected{});
  REQUIRE(thrust::equal(d_out.cbegin(), d_out.cend(), expected));
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
  SUCCEED("exceeding memory is not a failure");
}
