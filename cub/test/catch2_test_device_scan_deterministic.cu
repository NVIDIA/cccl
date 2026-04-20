// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_scan.cuh>

#include <thrust/equal.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/functional>

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
    1, // hits small copy path for bulk copies (below 16 bytes)
    10,
    1337,
    3000,
    1 * 31 * 128, // tile size for int64s for lookahead
    10'000, // a handful of tiles for lookahead
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const int num_runs = GENERATE(2, 5, 10);
  CAPTURE(num_items, num_runs);

  // use reasonable magnitude to avoid overflow
  constexpr type max_mag = type{100};
  c2h::device_vector<type> d_input(num_items);
  c2h::gen(C2H_SEED(1), d_input, -max_mag, max_mag);

  const auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);
  const auto op  = cuda::std::plus<type>{};

  c2h::device_vector<type> d_reference(num_items);
  REQUIRE(
    cub::DeviceScan::ExclusiveScan(d_input.begin(), d_reference.begin(), op, type{}, num_items, env) == cudaSuccess);

  c2h::device_vector<type> d_output(num_items);
  for (int run = 1; run < num_runs; ++run)
  {
    REQUIRE(
      cub::DeviceScan::ExclusiveScan(d_input.begin(), d_output.begin(), op, type{}, num_items, env) == cudaSuccess);
    REQUIRE(thrust::equal(d_output.begin(), d_output.end(), d_reference.begin()));
  }
}
