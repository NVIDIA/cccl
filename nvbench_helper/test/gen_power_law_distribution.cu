// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>

#include <nvbench_helper.cuh>

#include <boost/math/statistics/anderson_darling.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

bool is_normal(thrust::host_vector<double> data)
{
  std::sort(data.begin(), data.end());
  const double A2 = boost::math::statistics::anderson_darling_normality_statistic(data);
  return A2 / data.size() < 0.05;
}

using types = nvbench::type_list<uint32_t, uint64_t>;

TEMPLATE_LIST_TEST_CASE("Generators produce power law distributed data", "[gen][power-law]", types)
{
  const std::size_t elements                              = 1 << 28;
  const std::size_t segments                              = 4 * 1024;
  const thrust::device_vector<TestType> d_segment_offsets = generate.power_law.segment_offsets(elements, segments);
  REQUIRE(d_segment_offsets.size() == segments + 1);

  std::size_t actual_elements = 0;
  thrust::host_vector<double> log_sizes(segments);
  const thrust::host_vector<TestType> h_segment_offsets = d_segment_offsets;
  for (std::size_t i = 0; i < segments; ++i)
  {
    const TestType begin = h_segment_offsets[i];
    const TestType end   = h_segment_offsets[i + 1];
    REQUIRE(begin <= end);

    const std::size_t size = end - begin;
    actual_elements += size;
    log_sizes[i] = std::log(size);
  }

  REQUIRE(actual_elements == elements);
  REQUIRE(is_normal(std::move(log_sizes)));
}
