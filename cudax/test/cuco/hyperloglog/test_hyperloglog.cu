//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/hash_functions.cuh>
#include <cuda/experimental/__cuco/hyperloglog.cuh>
#include <cuda/experimental/__cuco/hyperloglog_ref.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

#include <c2h/catch2_test_helper.h>

namespace cudax = cuda::experimental;

template <typename Ref, typename InputIt, typename OutputIt>
__global__ void estimate_kernel(typename Ref::sketch_size_kb sketch_size_kb, InputIt in, size_t n, OutputIt out)
{
  extern __shared__ cuda::std::byte local_sketch[];

  const auto block = cooperative_groups::this_thread_block();

  // only a single block computes the estimate
  if (block.group_index().x == 0)
  {
    Ref estimator(cuda::std::span(local_sketch, Ref::sketch_bytes(sketch_size_kb)));

    estimator.clear(block);
    block.sync();

    for (int i = block.thread_rank(); i < n; i += block.num_threads())
    {
      estimator.add(*(in + i));
    }
    block.sync();
    const auto estimate = estimator.estimate(block);
    if (block.thread_rank() == 0)
    {
      *out = estimate;
    }
  }
}

using test_types = c2h::type_list<int32_t, int64_t>;

C2H_TEST("HyperLogLog device ref", "[hyperloglog]", test_types)
{
  using T              = c2h::get<0, TestType>;
  using estimator_type = cudax::cuco::hyperloglog<T>;

  // Test parameters
  const std::size_t num_items_pow2 = GENERATE(25, 26, 28);
  const int hll_precision          = GENERATE(8, 10, 12, 13);
  const typename estimator_type::sketch_size_kb sketch_size_kb(4.0 * (1ull << hll_precision) / 1024.0);
  const std::size_t num_items = 1ull << num_items_pow2;

  CAPTURE(num_items, hll_precision, sketch_size_kb);

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), T{0});

  // Initialize the estimator
  estimator_type estimator{sketch_size_kb};

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  const auto host_estimate = estimator.estimate();

  thrust::device_vector<std::size_t> device_estimate(1);
  estimate_kernel<typename estimator_type::template ref_type<cuda::thread_scope_block>>
    <<<1, 512, estimator.sketch_bytes()>>>(sketch_size_kb, items.begin(), num_items, device_estimate.begin());

  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::size_t device_estimate_value = device_estimate[0];
  REQUIRE(device_estimate_value == host_estimate);
}

C2H_TEST("HyperLogLog unique sequence", "[hyperloglog]", test_types)
{
  using T              = c2h::get<0, TestType>;
  using estimator_type = cudax::cuco::hyperloglog<T>;

  const std::size_t num_items_pow2 = GENERATE(25, 26, 28);
  const int hll_precision          = GENERATE(8, 10, 12, 13, 18);
  const typename estimator_type::sketch_size_kb sketch_size_kb(4.0 * (1ull << hll_precision) / 1024.0);
  const std::size_t num_items = 1ull << num_items_pow2;

  CAPTURE(num_items, hll_precision, sketch_size_kb);

  // This factor determines the error threshold for passing the test
  constexpr double tolerance_factor = 2.5;
  // RSD for a given precision is given by the following formula
  const double relative_standard_deviation = 1.04 / std::sqrt(static_cast<double>(1ull << hll_precision));

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), T{0});

  // Initialize the estimator
  estimator_type estimator{sketch_size_kb};

  REQUIRE(estimator.estimate() == 0);

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  const auto estimate = estimator.estimate();

  // Adding the same items again should not affect the result
  estimator.add(items.begin(), items.begin() + num_items / 2);
  REQUIRE(estimator.estimate() == estimate);

  // Clearing the estimator should reset the estimate
  estimator.clear();
  REQUIRE(estimator.estimate() == 0);

  const double relative_error = std::abs((static_cast<double>(estimate) / static_cast<double>(num_items)) - 1.0);

  // Check if the error is acceptable
  REQUIRE(relative_error < tolerance_factor * relative_standard_deviation);
}

//! @brief The following unit tests mimic Spark's unit tests which can be found here:
//! https://github.com/apache/spark/blob/d10dbaa31a44878df5c7e144f111e18261346531/sql/catalyst/src/test/scala/org/apache/spark/sql/catalyst/expressions/aggregate/HyperLogLogPlusPlusSuite.scala
//!

C2H_TEST("HyperLogLog Spark parity deterministic", "[hyperloglog]")
{
  using T              = int;
  using estimator_type = cudax::cuco::hyperloglog<T>;

  constexpr std::size_t repeats = 10;
  // This factor determines the error threshold for passing the test
  constexpr double tolerance_factor = 3.0;
  const auto num_items              = GENERATE(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000);
  const auto standard_deviation     = GENERATE(0.1, 0.05, 0.025, 0.01, 0.005, 0.0025);

  const auto expected_hll_precision =
    std::max(static_cast<int32_t>(4),
             static_cast<int32_t>(std::ceil(2.0 * std::log(1.106 / standard_deviation) / std::log(2.0))));
  const auto expected_sketch_bytes = 4 * (1ull << expected_hll_precision);

  CAPTURE(num_items, standard_deviation, expected_hll_precision, expected_sketch_bytes);

  const estimator_type::standard_deviation sd(standard_deviation);
  const estimator_type::sketch_size_kb sb(expected_sketch_bytes / 1024.0);

  // Validate sketch size calculation
  REQUIRE(estimator_type::sketch_bytes(sd) >= 64);
  REQUIRE(estimator_type::sketch_bytes(sd) == expected_sketch_bytes);
  REQUIRE(estimator_type::sketch_bytes(sd) == estimator_type::sketch_bytes(sb));

  auto items_begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator<std::size_t>(0), cuda::proclaim_return_type<T>([repeats] __device__(auto i) {
      return static_cast<T>(i / repeats);
    }));

  estimator_type estimator{sd};

  REQUIRE(estimator.estimate() == 0);

  // Add all items to the estimator
  estimator.add(items_begin, items_begin + num_items);

  const auto estimate = estimator.estimate();

  const double relative_error =
    std::abs((static_cast<double>(estimate) / static_cast<double>(num_items / repeats)) - 1.0);
  // RSD for a given precision is given by the following formula
  const double expected_standard_deviation = 1.04 / std::sqrt(static_cast<double>(1ull << expected_hll_precision));

  // Check if the error is acceptable
  REQUIRE(relative_error < expected_standard_deviation * tolerance_factor);
}

C2H_TEST("HyperLogLog precision constructor", "[hyperloglog]")
{
  using T              = int;
  using estimator_type = cudax::cuco::hyperloglog<T>;

  const auto precision_value = GENERATE(4, 6, 8, 12, 16, 18);

  const estimator_type::precision precision(precision_value);
  const auto expected_sketch_bytes = 4 * (1ull << precision_value);

  CAPTURE(precision_value, expected_sketch_bytes);

  REQUIRE(estimator_type::sketch_bytes(precision) == expected_sketch_bytes);

  estimator_type estimator{precision};

  REQUIRE(estimator.sketch_bytes() == expected_sketch_bytes);
  REQUIRE(estimator.estimate() == 0);
}

#if _CCCL_CTK_AT_LEAST(12, 6) // Pinned memory resource is only supported with CTK 12.6 and later
C2H_TEST("Hyperloglog estimate works with pinned memory pool", "[hyperloglog]")
{
  using T              = int32_t;
  using estimator_type = cudax::cuco::hyperloglog<T>;

  const std::size_t num_items = 1 << 20;
  const int hll_precision     = 12;
  const typename estimator_type::sketch_size_kb sketch_size_kb(4.0 * (1ull << hll_precision) / 1024.0);

  CAPTURE(num_items, hll_precision, sketch_size_kb);

  constexpr double tolerance_factor        = 2.5;
  const double relative_standard_deviation = 1.04 / std::sqrt(static_cast<double>(1ull << hll_precision));

  thrust::device_vector<T> items(num_items);
  thrust::sequence(items.begin(), items.end(), T{0});

  estimator_type estimator{sketch_size_kb};
  estimator.add(items.begin(), items.end());

  auto host_mr        = ::cuda::pinned_default_memory_pool();
  const auto estimate = estimator.estimate(host_mr);

  const double relative_error = std::abs((static_cast<double>(estimate) / static_cast<double>(num_items)) - 1.0);

  REQUIRE(relative_error < tolerance_factor * relative_standard_deviation);
}
#endif // _CCCL_CTK_AT_LEAST(12, 6)
