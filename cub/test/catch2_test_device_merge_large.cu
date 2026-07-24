// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Large-memory DeviceMerge tests, split out from catch2_test_device_merge.cu so they can be run
// serially (tagged [large-mem]) without serializing the small tests in that file.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_merge.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <cuda/iterator>

#include <algorithm>

#include <test_util.h>

#include "catch2_test_device_merge_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

C2H_TEST("DeviceMerge::MergeKeys works for large number of items",
         "[large-mem][merge][device][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]")
try
{
  using key_t    = char;
  using offset_t = int64_t;

  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const auto num_items_int_max = static_cast<offset_t>(cuda::std::numeric_limits<std::int32_t>::max());

  // Generate the input sizes to test for
  const offset_t num_items_lhs =
    GENERATE_COPY(values({num_items_int_max + offset_t{1000000}, num_items_int_max - 1, offset_t{3}}));
  const offset_t num_items_rhs =
    GENERATE_COPY(values({num_items_int_max + offset_t{1000000}, num_items_int_max, offset_t{3}}));

  test_keys<key_t, offset_t>(num_items_lhs, num_items_rhs, cuda::std::less<>{});
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
  SUCCEED("allocation failure is not a test failure");
}

C2H_TEST("DeviceMerge::MergePairs really large input",
         "[large-mem][merge][device][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]")
try
{
  using key_t     = char;
  using value_t   = char;
  const auto size = std::int64_t{1} << GENERATE(30, 31, 32, 33);
  test_pairs<key_t, value_t>(size, size, cuda::std::less<>{});
}
catch (const std::bad_alloc&)
{
  // allocation failure is not a test failure, so we can run tests on smaller GPUs
  SUCCEED("allocation failure is not a test failure");
}
