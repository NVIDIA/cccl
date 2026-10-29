// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% TEST_ERR err 0:1:2:3

#include <cub/device/device_reduce.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/functional>

// Verifies that a tuning selecting a non-deterministic block reduce is still rejected if the reduction is required to
// be (or defaults to) run-to-run deterministic. Only the multi-tile policy is non-deterministic, so each test case
// triggers a single assertion.

struct nondeterministic_reduce_tuning
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability) const -> cub::ReducePolicy
  {
    const auto multi_tile =
      cub::ReducePassPolicy{128, 4, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC, cub::LOAD_DEFAULT};
    const auto single_tile = cub::ReducePassPolicy{128, 4, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {multi_tile, single_tile};
  }
};

int main()
{
  float* in{};
  float* out{};
  float* max_out{};
  int* index_out{};
  int* max_index_out{};

  auto run_to_run_env = cuda::std::execution::env{cuda::execution::require(cuda::execution::determinism::run_to_run),
                                                  cuda::execution::tune(nondeterministic_reduce_tuning{})};
  auto default_env    = cuda::execution::tune(nondeterministic_reduce_tuning{});

#if TEST_ERR == 0
  // expected-error-0 {{"A run-to-run deterministic reduction must not use a non-deterministic reduce_algorithm"}}
  auto error = cub::DeviceReduce::Reduce(in, out, 0, cuda::maximum<>{}, 0.0f, run_to_run_env);
#elif TEST_ERR == 1
  // expected-error-1 {{"A run-to-run deterministic reduction must not use a non-deterministic reduce_algorithm"}}
  auto error = cub::DeviceReduce::Reduce(in, out, 0, cuda::maximum<>{}, 0.0f, default_env);
#elif TEST_ERR == 2
  // Min and Max are always run-to-run deterministic, unless the user requests not_guaranteed
  // expected-error-2 {{"A run-to-run deterministic reduction must not use a non-deterministic reduce_algorithm"}}
  auto error = cub::DeviceReduce::Min(in, out, 0, default_env);
#elif TEST_ERR == 3
  // ArgMin, ArgMax and ArgMinMax are always run-to-run deterministic, unless the user requests not_guaranteed
  // expected-error-3 {{"A run-to-run deterministic reduction must not use a non-deterministic reduce_algorithm"}}
  auto error = cub::DeviceReduce::ArgMin(in, out, index_out, 0, default_env);
#endif
  (void) max_out;
  (void) max_index_out;
  return static_cast<int>(error);
}
