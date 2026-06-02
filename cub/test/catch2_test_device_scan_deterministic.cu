// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/functional>

#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/generators.h>

using float_type_list = c2h::type_list<float, double>;

C2H_TEST("DeviceScan::ExclusiveScan with run_to_run determinism is bit-reproducible",
         "[scan][deterministic][run_to_run]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int min_items = 1;
  constexpr int max_items = 1 << 22;

  // Generate the input sizes to test for
  const int num_items = GENERATE_COPY(
    1337,
    3000,
    1 * 31 * 128, // tile size for int64s for lookahead
    10'000, // a handful of tiles for lookahead
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const int num_runs = 10;
  CAPTURE(num_items, num_runs);

  // use reasonable magnitude to avoid overflow
  constexpr type max_mag = type{100};
  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_input, -max_mag, max_mag);

  const auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  const auto op  = cuda::std::plus<type>{};

  c2h::device_vector<type> d_reference(num_items);
  REQUIRE(
    cub::DeviceScan::ExclusiveScan(d_input.begin(), d_reference.begin(), op, type{}, num_items, env) == cudaSuccess);

  c2h::device_vector<type> d_output(num_items, thrust::no_init);
  for (int run = 1; run < num_runs; ++run)
  {
    REQUIRE(
      cub::DeviceScan::ExclusiveScan(d_input.begin(), d_output.begin(), op, type{}, num_items, env) == cudaSuccess);

    // Verify bitwise equality of the reference and output results for every run
    REQUIRE_THAT(detail::to_vec(d_reference), detail::BitwiseEqualsRange(detail::to_vec(d_output)));
  }
}

C2H_TEST("DeviceScan::ExclusiveScan with run_to_run determinism matches host reference",
         "[scan][deterministic][run_to_run]",
         float_type_list)
{
  using type = typename c2h::get<0, TestType>;

  constexpr int min_items = 1;
  constexpr int max_items = 1 << 22;

  const int num_items = GENERATE_COPY(
    1337,
    3000,
    1 * 31 * 128,
    10'000,
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));
  CAPTURE(num_items);

  constexpr type max_mag = type{100};
  c2h::device_vector<type> d_input(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(1), d_input, type{0}, max_mag);

  const auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  const auto op  = cuda::std::plus<type>{};

  c2h::host_vector<type> h_input(d_input);
  c2h::host_vector<type> h_reference(num_items);
  compute_exclusive_scan_reference(h_input.cbegin(), h_input.cend(), h_reference.begin(), type{}, op);

  c2h::device_vector<type> d_output(num_items);
  REQUIRE(cub::DeviceScan::ExclusiveScan(d_input.begin(), d_output.begin(), op, type{}, num_items, env) == cudaSuccess);

  REQUIRE_APPROX_EQ_EPSILON(h_reference, d_output, type{0.05});
}
