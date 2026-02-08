// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/equal.h>

#include <nvbench_helper.cuh>

#include <catch2/catch_template_test_macros.hpp>

using types =
  nvbench::type_list<bool,
                     int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#if NVBENCH_HELPER_HAS_I128
                     int128_t,
#endif
                     float,
                     double,
                     complex>;

TEMPLATE_LIST_TEST_CASE("Generator seeds the data", "[gen]", types)
{
  auto generator = generate(1 << 24, bit_entropy::_0_811);

  const thrust::device_vector<TestType> vec_1 = generator;
  const thrust::device_vector<TestType> vec_2 = generator;

  REQUIRE(vec_1.size() == vec_2.size());
  REQUIRE_FALSE(thrust::equal(vec_1.begin(), vec_1.end(), vec_2.begin()));
}
