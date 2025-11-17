//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
__global__ void estimate_kernel(double sketch_size_kb, InputIt in, size_t n, OutputIt out)
{
  extern __shared__ cuda::std::byte local_sketch[];

  auto const block = cooperative_groups::this_thread_block();

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
    auto const estimate = estimator.estimate(block);
    if (block.thread_rank() == 0)
    {
      *out = estimate;
    }
  }
}

using test_types = c2h::type_list<int32_t, int64_t, __int128_t>;

C2H_TEST("HyperLogLog device ref", "[hyperloglog]", test_types)
{
  using T              = c2h::get<0, TestType>;
  using estimator_type = cudax::cuco::hyperloglog<T>;

  // Test parameters
  std::size_t num_items_pow2 = GENERATE(25, 26, 28);
  int hll_precision          = GENERATE(8, 10, 12, 13);
  double sketch_size_kb      = 4.0 * (1ull << hll_precision) / 1024.0;
  std::size_t num_items      = 1ull << num_items_pow2;

  CAPTURE(num_items, hll_precision, sketch_size_kb);

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), T{0});

  // Initialize the estimator
  estimator_type estimator{sketch_size_kb};

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  auto const host_estimate = estimator.estimate();

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

  std::size_t num_items_pow2 = GENERATE(25, 26, 28);
  int hll_precision          = GENERATE(8, 10, 12, 13, 18, 20);
  double sketch_size_kb      = 4.0 * (1ull << hll_precision) / 1024.0;
  std::size_t num_items      = 1ull << num_items_pow2;

  CAPTURE(num_items, hll_precision, sketch_size_kb);

  // This factor determines the error threshold for passing the test
  double constexpr tolerance_factor = 2.5;
  // RSD for a given precision is given by the following formula
  double const relative_standard_deviation = 1.04 / std::sqrt(static_cast<double>(1ull << hll_precision));

  thrust::device_vector<T> items(num_items);

  // Generate `num_items` distinct items
  thrust::sequence(items.begin(), items.end(), T{0});

  // Initialize the estimator
  estimator_type estimator{sketch_size_kb};

  REQUIRE(estimator.estimate() == 0);

  // Add all items to the estimator
  estimator.add(items.begin(), items.end());

  auto const estimate = estimator.estimate();

  // Adding the same items again should not affect the result
  estimator.add(items.begin(), items.begin() + num_items / 2);
  REQUIRE(estimator.estimate() == estimate);

  // Clearing the estimator should reset the estimate
  estimator.clear();
  REQUIRE(estimator.estimate() == 0);

  double const relative_error = std::abs((static_cast<double>(estimate) / static_cast<double>(num_items)) - 1.0);

  // Check if the error is acceptable
  REQUIRE(relative_error < tolerance_factor * relative_standard_deviation);
}
