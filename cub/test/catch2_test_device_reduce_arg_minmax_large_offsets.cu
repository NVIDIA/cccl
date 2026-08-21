// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <cuda/iterator>

#include <cstdint>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinMax, device_arg_minmax);
DECLARE_LAUNCH_WRAPPER(cub::DeviceReduce::ArgMinLastMax, device_arg_minlastmax);

// %PARAM% TEST_LAUNCH lid 0:1:2

// List of offset types to test
using offset_types = c2h::type_list<std::int32_t, std::uint32_t, std::uint64_t>;

// Dispatches to ArgMinMax (first maximum) or ArgMinLastMax (last maximum) so a single test body covers both APIs.
template <typename... Args>
void call_argminmax_launch_wrapper(bool last_max, Args... args)
{
  if (last_max)
  {
    device_arg_minlastmax(args...);
  }
  else
  {
    device_arg_minmax(args...);
  }
}

// Maps an index to a value that is the global minimum (0) at two positions, the global maximum (2) at two other
// positions, and a filler (1) everywhere else. Repeating the extrema lets us place one occurrence near the start and
// one near the end of the input, so for large offset types they land in different streaming partitions.
template <typename ItemT>
struct repeated_extrema_op
{
  uint64_t min_pos_first;
  uint64_t min_pos_last;
  uint64_t max_pos_first;
  uint64_t max_pos_last;

  __host__ __device__ _CCCL_FORCEINLINE ItemT operator()(const uint64_t index) const
  {
    if (index == min_pos_first || index == min_pos_last)
    {
      return ItemT{0};
    }
    if (index == max_pos_first || index == max_pos_last)
    {
      return ItemT{2};
    }
    return ItemT{1};
  }
};

CUB_TEST(
  "Device ArgMin[Last]Max reports first/last extremum across partitions", "[reduce][device]", CUB_SMALL, offset_types)
{
  using index_t  = uint64_t;
  using offset_t = typename c2h::get<0, TestType>;
  using item_t   = int;

  CAPTURE(c2h::type_name<offset_t>());

  const bool last_max = GENERATE(false, true);

  // Large enough that 32/64-bit offset types span more than one streaming partition.
  const offset_t num_items = detail::make_large_offset<offset_t>();
  const index_t n          = static_cast<index_t>(num_items);

  // Place the repeated minimum (0) and maximum (2) once near the start and once near the end of the input, so their two
  // occurrences fall into different partitions. This verifies that, across the partition boundary, the minimum and
  // ArgMinMax's maximum resolve to the first occurrence while ArgMinLastMax's maximum resolves to the last.
  const index_t min_pos_first = 0;
  const index_t min_pos_last  = n - 2;
  const index_t max_pos_first = 1;
  const index_t max_pos_last  = n - 1;

  const auto d_in_it = cuda::transform_iterator(
    cuda::counting_iterator(index_t{}),
    repeated_extrema_op<item_t>{min_pos_first, min_pos_last, max_pos_first, max_pos_last});

  c2h::device_vector<item_t> min_out(1, thrust::no_init);
  c2h::device_vector<item_t> max_out(1, thrust::no_init);
  c2h::device_vector<cuda::std::int64_t> min_index(1, thrust::no_init);
  c2h::device_vector<cuda::std::int64_t> max_index(1, thrust::no_init);

  call_argminmax_launch_wrapper(
    last_max,
    d_in_it,
    thrust::raw_pointer_cast(min_out.data()),
    thrust::raw_pointer_cast(min_index.data()),
    thrust::raw_pointer_cast(max_out.data()),
    thrust::raw_pointer_cast(max_index.data()),
    num_items);

  const auto expected_max_index = static_cast<cuda::std::int64_t>(last_max ? max_pos_last : max_pos_first);

  REQUIRE(min_out[0] == item_t{0});
  REQUIRE(min_index[0] == static_cast<cuda::std::int64_t>(min_pos_first));
  REQUIRE(max_out[0] == item_t{2});
  REQUIRE(max_index[0] == expected_max_index);
}
