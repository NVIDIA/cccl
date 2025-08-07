// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/grid/grid_even_share.cuh>
#include <cub/grid/grid_mapping.cuh>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/generators.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

using offset_types = c2h::type_list<int32_t, int64_t, uint32_t, uint64_t>;

C2H_TEST("GridEvenShare handles edge cases (zero/negative items)", "[grid][even_share][edge_cases]", offset_types)
{
  using offset_t = typename c2h::get<0, TestType>;

  cub::GridEvenShare<offset_t> grid_share;

  const offset_t num_items = []() {
    if constexpr (cuda::std::is_signed_v<offset_t>)
    {
      return GENERATE_COPY(values({-1, 0, 1}));
    }
    else
    {
      return GENERATE_COPY(values({0, 1}));
    }
  }();

  const int max_grid_size = GENERATE_COPY(values({-1, 0, 1}));
  const int tile_items    = GENERATE_COPY(values({-1, 0, 1}));

  // Skip if all parameters are positive (covered by the normal operation test)
  if (num_items > 0 && max_grid_size > 0 && tile_items > 0)
  {
    return;
  }

  grid_share.DispatchInit(num_items, max_grid_size, tile_items);

  REQUIRE(grid_share.num_items == 0);
  REQUIRE(grid_share.grid_size == 0);
  REQUIRE(grid_share.block_offset == 0);
  REQUIRE(grid_share.block_end == 0);
}

C2H_TEST("GridEvenShare works with num_items > 0", "[grid][even_share]", offset_types)
{
  using offset_t = typename c2h::get<0, TestType>;

  cub::GridEvenShare<offset_t> grid_share;

  const offset_t num_items = GENERATE_COPY(values({1, 20, 37, 100, 2000, 1 << 20}));
  const int max_grid_size  = GENERATE_COPY(values({1, 20, 37, 100, 2000, 1 << 20}));
  const int tile_items     = GENERATE_COPY(values({1, 20, 37, 100, 2000, 1 << 20}));

  grid_share.DispatchInit(num_items, max_grid_size, tile_items);

  REQUIRE(grid_share.num_items == num_items);
  REQUIRE(
    grid_share.grid_size == cuda::std::min(max_grid_size, static_cast<int>(cuda::ceil_div(num_items, tile_items))));
  REQUIRE(grid_share.block_offset == num_items);
  REQUIRE(grid_share.block_end == num_items);
}
