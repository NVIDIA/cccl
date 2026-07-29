// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Large-memory DeviceFind tests, split out from catch2_test_device_find.cu so they can be
// run serially (tagged [large-mem]) without serializing the small tests in that file.

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>

#include <cuda/iterator>

#include "catch2_test_device_find_common.cuh"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBound, lower_bound);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBound, upper_bound);

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceFind::LowerBound really large input",
         "[large-mem][find][device][binary-search][skip-cs-rangecheck][skip-cs-initcheck][skip-cs-synccheck]")
{
  try
  {
    using value_type = char;
    const auto size  = std::int64_t{1} << GENERATE(30, 31, 32, 33);
    test_vectorized<value_type>(lower_bound, std_lower_bound, size);
  }
  catch (const std::bad_alloc&)
  {
    // allocation failure is not a test failure, so we can run tests on smaller GPUs
    SUCCEED("allocation failure is not a test failure");
  }
}

// this test exceeds 4GiB of memory and the range of 32-bit integers
C2H_TEST("DeviceFind::UpperBound really large input",
         "[large-mem][find][device][binary-search][skip-cs-rangecheck][skip-cs-initcheck][skip-cs-synccheck]")
{
  try
  {
    using value_type = char;
    const auto size  = std::int64_t{1} << GENERATE(30, 31, 32, 33);
    test_vectorized<value_type>(upper_bound, std_upper_bound, size);
  }
  catch (const std::bad_alloc&)
  {
    // allocation failure is not a test failure, so we can run tests on smaller GPUs
    SUCCEED("allocation failure is not a test failure");
  }
}
