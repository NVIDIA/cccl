// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <cuda/iterator>

#include <cstdint>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::ExclusiveScan, device_exclusive_scan);
DECLARE_LAUNCH_WRAPPER(cub::DeviceScan::InclusiveScan, device_inclusive_scan);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of offset types to be used for testing large number of items
using offset_types = c2h::type_list<std::uint32_t, std::uint64_t>;

template <typename ItemT>
struct expected_exclusive_sum_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ ItemT operator()(const uint64_t index) const
  {
    uint64_t sum_per_full_segment = (segment_size * (segment_size - 1)) / 2;
    uint64_t full_segments        = index / segment_size;
    uint64_t index_within_segment = index % segment_size;

    uint64_t sum_within_partial_segment = (index_within_segment * (index_within_segment - 1)) / 2;
    uint64_t sum_over_full_segments     = full_segments * sum_per_full_segment;
    return static_cast<ItemT>(sum_within_partial_segment + sum_over_full_segments);
  }
};

template <typename ItemT>
struct expected_inclusive_sum_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ ItemT operator()(const uint64_t index) const
  {
    // The sum of a completed segment (0 to segment_size-1)
    // Formula: n(n+1)/2 where n is segment_size-1
    uint64_t sum_per_full_segment = (segment_size * (segment_size - 1)) / 2;

    uint64_t full_segments        = index / segment_size;
    uint64_t index_within_segment = index % segment_size;

    // For inclusive, this includes the current index value in the sum.
    uint64_t sum_within_partial_segment = (index_within_segment * (index_within_segment + 1)) / 2;

    uint64_t sum_over_full_segments = full_segments * sum_per_full_segment;

    return static_cast<ItemT>(sum_within_partial_segment + sum_over_full_segments);
  }
};

template <typename ItemT>
struct mod_op
{
  uint64_t segment_size;

  __host__ __device__ __forceinline__ uint64_t operator()(const uint64_t index) const
  {
    return static_cast<ItemT>(index % segment_size);
  }
};

C2H_TEST("DeviceScan works for very large number of items", "[scan][device]", offset_types)
try
{
  using op_t     = cuda::std::plus<>;
  using item_t   = std::uint32_t;
  using index_t  = std::uint64_t;
  using offset_t = typename c2h::get<0, TestType>;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const offset_t num_items_max = detail::make_large_offset<offset_t>();
  const offset_t num_items_min = num_items_max > 10000 ? num_items_max - 10000ULL : offset_t{0};
  const offset_t num_items     = GENERATE_COPY(
    values(
      {num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1), static_cast<offset_t>(3)}),
    take(2, random(num_items_min, num_items_max)));

  // Prepare input (generate a series of: 0, 1, 2, ..., <segment_size-1>,  0, 1, 2, ..., <segment_size-1>, 0, 1, ...)
  constexpr index_t segment_size = 1000;
  auto index_it                  = cuda::counting_iterator(index_t{});
  auto items_it                  = cuda::transform_iterator(index_it, mod_op<item_t>{segment_size});

  // Output memory allocation
  c2h::device_vector<item_t> d_items_out(num_items);
  auto d_items_out_it = thrust::raw_pointer_cast(d_items_out.data());

  c2h::device_vector<item_t> d_initial_value(1);
  d_initial_value[0]     = item_t{};
  auto future_init_value = cub::FutureValue<item_t>(thrust::raw_pointer_cast(d_initial_value.data()));

  // Run test
  device_exclusive_scan(items_it, d_items_out_it, op_t{}, future_init_value, num_items);

  // Ensure that we created the correct output
  auto expected_out_it =
    cuda::transform_iterator(index_it, expected_exclusive_sum_op<item_t>{static_cast<index_t>(segment_size)});
  bool all_results_correct = thrust::equal(d_items_out.cbegin(), d_items_out.cend(), expected_out_it);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}

C2H_TEST("DeviceScan works for very large number of items", "[scan][device]", offset_types)
try
{
  using op_t     = cuda::std::plus<>;
  using item_t   = int;
  using index_t  = std::int64_t;
  using offset_t = typename c2h::get<0, TestType>;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const offset_t num_items_max = detail::make_large_offset<offset_t>();
  const offset_t num_items_min = num_items_max > 10000 ? num_items_max - 10000ULL : offset_t{0};
  const offset_t num_items     = GENERATE_COPY(
    values(
      {num_items_max, static_cast<offset_t>(num_items_max - 1), static_cast<offset_t>(1), static_cast<offset_t>(3)}),
    take(2, random(num_items_min, num_items_max)));

  CAPTURE(num_items);

  // Prepare input (generate a series of: 0, 1, 2, ..., <segment_size-1>,  0, 1, 2, ..., <segment_size-1>, 0, 1, ...)
  constexpr index_t segment_size = 1000;
  auto index_it                  = cuda::counting_iterator(index_t{});
  auto items_it                  = cuda::transform_iterator(index_it, mod_op<item_t>{segment_size});
  c2h::device_vector<item_t> d_items_in(items_it, items_it + num_items);
  auto d_items_ptr = thrust::raw_pointer_cast(d_items_in.data());

  // Output memory allocation
  c2h::device_vector<item_t> d_items_out(num_items);
  auto d_items_out_it = thrust::raw_pointer_cast(d_items_out.data());

  // Run test
  device_inclusive_scan(d_items_ptr, d_items_out_it, op_t{}, num_items);

  // Ensure that we created the correct output
  auto expected_out_it =
    cuda::transform_iterator(index_it, expected_inclusive_sum_op<item_t>{static_cast<index_t>(segment_size)});
  bool all_results_correct = thrust::equal(d_items_out.cbegin(), d_items_out.cend(), expected_out_it);
  REQUIRE(all_results_correct == true);
}
catch (std::bad_alloc&)
{
  // Exceeding memory is not a failure.
}
