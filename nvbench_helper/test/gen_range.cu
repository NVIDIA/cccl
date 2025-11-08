// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <limits>

#include <nvbench_helper.cuh>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

using types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if NVBENCH_HELPER_HAS_I128
                     int128_t,
#endif
                     float,
                     double>;

TEMPLATE_LIST_TEST_CASE("Generators produce data within specified range", "[gen]", types)
{
  const auto min = static_cast<TestType>(GENERATE_COPY(take(3, random(-124, 0))));
  const auto max = static_cast<TestType>(GENERATE_COPY(take(3, random(0, 124))));

  const thrust::device_vector<TestType> data = generate(1 << 16, bit_entropy::_1_000, min, max);

  const TestType min_element = *thrust::min_element(data.begin(), data.end());
  const TestType max_element = *thrust::max_element(data.begin(), data.end());

  REQUIRE(min_element >= min);
  REQUIRE(max_element <= max);
}
