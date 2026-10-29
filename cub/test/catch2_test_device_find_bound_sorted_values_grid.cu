// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes.
struct limited_grid_factory;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY limited_grid_factory

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <cuda/iterator>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

// Exercise multiple search and partition launches without allocating billions of partition offsets.
struct limited_grid_factory : cub::detail::TripleChevronFactory
{
  CUB_RUNTIME_FUNCTION cudaError_t MaxGridDimX(int& max_grid_dim_x) const
  {
    max_grid_dim_x = 2;
    return cudaSuccess;
  }
};

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBoundSortedValues, lower_bound_sorted_values);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBoundSortedValues, upper_bound_sorted_values);

struct scaled_index
{
  std::uint64_t stride;

  __host__ __device__ std::uint64_t operator()(std::uint64_t index) const
  {
    return index * stride;
  }
};

using offset_types = c2h::type_list<std::uint32_t, std::uint64_t>;

CUB_TEST("DeviceFind sorted bounds span multiple grids", "[find][device][binary-search]", CUB_SMALL, offset_types)
{
  using offset_type      = c2h::get<0, TestType>;
  using value_type       = std::uint64_t;
  const bool upper_bound = GENERATE(false, true);
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  const auto policy = cub::detail::find_bound_sorted_values::policy_selector_from_types<value_type, value_type>{}(cc);
  const auto tile_size = static_cast<offset_type>(policy.threads_per_block) * policy.items_per_thread;

  // Two blocks per grid: the search pass crosses a grid boundary at three tiles, while the
  // partition pass crosses one at 512 tiles (513 diagonals, with 256 diagonals per block).
  const offset_type num_tiles        = GENERATE(1, 2, 3, 511, 512, 513);
  const offset_type total_items      = num_tiles * tile_size - GENERATE(0, 1);
  const offset_type values_num_items = num_tiles;
  const offset_type range_num_items  = total_items - values_num_items;
  CAPTURE(upper_bound, num_tiles, total_items, tile_size);

  const auto range = cuda::counting_iterator<value_type>{0};
  // Spread needles across all tiles so that omitting any search grid leaves outputs unwritten.
  const auto values = cuda::make_transform_iterator(range, scaled_index{tile_size - 1});
  c2h::device_vector<value_type> offsets_d(values_num_items, std::numeric_limits<value_type>::max());
  const auto run = [&](auto variant) {
    variant(range,
            range_num_items,
            values,
            values_num_items,
            thrust::raw_pointer_cast(offsets_d.data()),
            cuda::std::less<value_type>{});
  };
  if (upper_bound)
  {
    run(upper_bound_sorted_values);
  }
  else
  {
    run(lower_bound_sorted_values);
  }

  c2h::host_vector<value_type> offsets_ref(values_num_items);
  for (offset_type i = 0; i < values_num_items; ++i)
  {
    const auto bound = upper_bound ? std::upper_bound(range, range + range_num_items, values[i])
                                   : std::lower_bound(range, range + range_num_items, values[i]);
    offsets_ref[i]   = static_cast<value_type>(bound - range);
  }
  REQUIRE(offsets_ref == offsets_d);
}
