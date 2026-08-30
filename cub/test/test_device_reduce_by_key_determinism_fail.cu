// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// %PARAM% TEST_ERR err 0:1

#include <cub/device/device_reduce.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/__functional/operations.h>

int main()
{
  int* keys{};
  double* values{};
  int* num_runs{};

#if TEST_ERR == 0
  // clang-format off
  // expected-error-0 {{"run_to_run deterministic reduce-by-key requires integral types with known operators, primitive types with min/max, or floating-point types with plus operator"}}
  // clang-format on
  auto error = cub::DeviceReduce::ReduceByKey(
    keys,
    keys,
    values,
    values,
    num_runs,
    cuda::std::multiplies<>{},
    0,
    cuda::execution::require(cuda::execution::determinism::run_to_run));
#else
  // clang-format off
  // expected-error-1 {{"gpu_to_gpu deterministic reduce-by-key requires integral types with known operators or primitive types with min/max"}}
  // clang-format on
  auto error = cub::DeviceReduce::ReduceByKey(
    keys,
    keys,
    values,
    values,
    num_runs,
    cuda::std::plus<>{},
    0,
    cuda::execution::require(cuda::execution::determinism::gpu_to_gpu));
#endif

  return error == cudaSuccess ? 0 : 1;
}
