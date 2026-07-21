// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// %PARAM% TEST_ERR err 0:1:2:3

#include <cub/device/device_scan.cuh>

#include <iostream>

int main()
{
  namespace stdexec = cuda::std::execution;

  float* ptr{};

#if TEST_ERR == 0
  // clang-format off
  // expected-error-0 {{"run_to_run deterministic scan requires either integral types with known operators, or floating-point types with plus operator"}}
  // clang-format on
  auto error = cub::DeviceScan::ExclusiveScan(
    ptr, ptr, cuda::std::multiplies<>{}, 0.0f, 0, cuda::execution::require(cuda::execution::determinism::run_to_run));
#elif TEST_ERR == 1
  // expected-error-1 {{"gpu_to_gpu deterministic scan requires integral types with known operators"}}
  auto error = cub::DeviceScan::ExclusiveScan(
    ptr, ptr, cuda::std::plus<>{}, 0.0f, 0, cuda::execution::require(cuda::execution::determinism::gpu_to_gpu));
#elif TEST_ERR == 2
  // clang-format off
  // expected-error-2 {{"run_to_run deterministic scan requires either integral types with known operators, or floating-point types with plus operator"}}
  // clang-format on
  auto future_init = cub::FutureValue<float>(ptr);
  auto error       = cub::DeviceScan::ExclusiveScan(
    ptr,
    ptr,
    cuda::std::multiplies<>{},
    future_init,
    0,
    cuda::execution::require(cuda::execution::determinism::run_to_run));
#elif TEST_ERR == 3
  // expected-error-3 {{"gpu_to_gpu deterministic scan requires integral types with known operators"}}
  auto future_init = cub::FutureValue<float>(ptr);
  auto error       = cub::DeviceScan::ExclusiveScan(
    ptr, ptr, cuda::std::plus<>{}, future_init, 0, cuda::execution::require(cuda::execution::determinism::gpu_to_gpu));
#endif

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan failed with status: " << error << '\n';
  }
}
