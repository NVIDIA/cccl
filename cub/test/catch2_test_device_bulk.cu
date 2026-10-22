// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::Bulk, device_bulk);

using offset_type = c2h::type_list<std::int32_t, std::uint32_t, std::uint64_t, std::int64_t>;

template <class T>
struct incrementer_t
{
  int* d_counts;

  template <class OffsetT>
  __device__ void operator()(OffsetT i)
  {
    static_assert(cuda::std::is_same_v<T, OffsetT>, "T and OffsetT must be the same type");
    atomicAdd(d_counts + i, 1); // Check if `i` was served more than once
  }
};

CUB_TEST("Device bulk works", "[bulk][device]", CUB_SMALL, offset_type)
{
  using offset_t = c2h::get<0, TestType>;

  constexpr int max_items = 5000000;
  constexpr int min_items = 1;

  const auto num_items = static_cast<offset_t>(GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    })));

  c2h::device_vector<int> counts(num_items);
  int* d_counts = thrust::raw_pointer_cast(counts.data());

  device_bulk(num_items, incrementer_t<offset_t>{d_counts});

  const auto num_of_once_marked_items =
    static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));

  REQUIRE(num_of_once_marked_items == num_items);
}

// Counts how often each index is visited and checks that no index outside [0, num_items) is passed to op
struct bounds_checking_incrementer_t
{
  int* d_counts;
  long long num_items;

  template <class IndexT>
  __device__ void operator()(IndexT i)
  {
    const auto idx = static_cast<long long>(i);
    REQUIRE(idx >= 0);
    REQUIRE(idx < num_items);
    atomicAdd(d_counts + idx, 1);
  }
};

// Tile size of 512 exceeds the range of 8-bit types
struct large_tile_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ForPolicy
  {
    return {128, 4};
  }
};

using narrow_offset_type = c2h::type_list<std::int8_t, std::uint8_t, std::int16_t, std::uint16_t>;

template <class OffsetT, class LaunchT>
void check_narrow_bulk(OffsetT num_items, LaunchT launch)
{
  c2h::device_vector<int> counts(num_items);
  const bounds_checking_incrementer_t op{thrust::raw_pointer_cast(counts.data()), static_cast<long long>(num_items)};

  launch(num_items, op);

  const auto num_of_once_marked_items = thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1);
  REQUIRE(num_of_once_marked_items == static_cast<decltype(num_of_once_marked_items)>(num_items));
}

CUB_TEST("Device bulk works with narrow shape types", "[bulk][device]", CUB_SMALL, narrow_offset_type)
{
  using offset_t = c2h::get<0, TestType>;

  const auto num_items = GENERATE(offset_t{1}, offset_t{100}, cuda::std::numeric_limits<offset_t>::max());

  check_narrow_bulk(num_items, [](offset_t n, bounds_checking_incrementer_t op) {
    device_bulk(n, op);
  });
}

#if TEST_LAUNCH == 0
CUB_TEST("Device bulk works with narrow shape types and tiles exceeding their range",
         "[bulk][device]",
         CUB_SMALL,
         narrow_offset_type)
{
  using offset_t = c2h::get<0, TestType>;

  const auto num_items = GENERATE(offset_t{1}, offset_t{100}, cuda::std::numeric_limits<offset_t>::max());

  check_narrow_bulk(num_items, [](offset_t n, bounds_checking_incrementer_t op) {
    REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(n, op, cuda::execution::tune(large_tile_policy_selector{})));
  });
}
#endif // TEST_LAUNCH == 0
