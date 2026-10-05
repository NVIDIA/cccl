// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/scan.h>
#include <thrust/system/omp/execution_policy.h>

#include <vector>

#include <omp.h>

#include <unittest/unittest.h>

TEST_CASE("OpenMP scans with a serialized nested team", "[scan]")
{
  constexpr int num_items = 4096;
  std::vector<int> input(num_items, 1), inclusive(num_items, -1), exclusive(num_items, -1);
  int outer_threads = 0;
  bool correct_ends = false;

#pragma omp parallel num_threads(2)
  {
#pragma omp single
    {
      outer_threads = omp_get_num_threads();
      // These settings are local to this implicit task, not the calling task.
      omp_set_dynamic(0);
      omp_set_nested(0);
      omp_set_num_threads(4);

      const auto inclusive_end =
        thrust::inclusive_scan(thrust::omp::par, input.begin(), input.end(), inclusive.begin());
      const auto exclusive_end =
        thrust::exclusive_scan(thrust::omp::par, input.begin(), input.end(), exclusive.begin(), 0);
      correct_ends = inclusive_end == inclusive.end() && exclusive_end == exclusive.end();
    }
  }

  if (outer_threads < 2)
  {
    SKIP("Two outer threads are required to serialize the nested scan team");
  }

  REQUIRE(correct_ends);
  for (int i = 0; i < num_items; ++i)
  {
    CHECK(inclusive[i] == i + 1);
    CHECK(exclusive[i] == i);
  }
}
