// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/type_traits>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

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

C2H_TEST("Device bulk works", "[bulk][device]", offset_type)
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
