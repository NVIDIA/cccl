// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Smoke test that a real algorithm dispatch emits a logging line when logging is enabled. CUB only relies on <iosfwd>,
// so the tuning policy is printed only if <sstream> is included by the user; otherwise it is reported as unavailable.

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <iostream>
#ifndef LOGGING_TEST_NO_SSTREAM
#  include <sstream> // required for CUB to print the tuning policy
#endif // !LOGGING_TEST_NO_SSTREAM

#include "cub_non_catch2_test_memory.h"

CUB_TEST_MEMORY_CLASS(CUB_SMALL);

int main()
{
  thrust::device_vector<int> in{1, 2, 3, 4, 5};
  thrust::device_vector<int> out(1, thrust::no_init);
  const cudaError_t status = cub::DeviceReduce::Sum(in.begin(), out.begin(), in.size());
  if (status != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::Sum failed: " << cudaGetErrorString(status) << '\n';
    return 1;
  }
  return out[0] == 15 ? 0 : 1;
}
