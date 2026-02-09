// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// %PARAM% TEST_ERR err 0:1

#include <cub/device/device_scan.cuh>

int main()
{
  namespace stdexec = cuda::std::execution;

  float* ptr{};

#if TEST_ERR == 0
  // expected-error {{"run_to_run or gpu_to_gpu is only supported for integral types with known operators"}}
  auto error = cub::DeviceScan::ExclusiveScan(
    ptr, ptr, cuda::std::plus<>{}, 0.0f, 0, cuda::execution::require(cuda::execution::determinism::run_to_run));
#elif TEST_ERR == 1
  // expected-error {{"run_to_run or gpu_to_gpu is only supported for integral types with known operators"}}
  auto error = cub::DeviceScan::ExclusiveScan(
    ptr, ptr, cuda::std::plus<>{}, 0.0f, 0, cuda::execution::require(cuda::execution::determinism::gpu_to_gpu));
#endif

  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceScan::ExclusiveScan failed with status: " << error << std::endl;
  }
}
