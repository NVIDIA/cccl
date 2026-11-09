// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <cuda/argument>

// %PARAM% TEST_ERR err 0:1:2:3

int main()
{
  int* items = nullptr;
#if TEST_ERR == 0
  const auto count = cuda::args::immediate<float>{0.0f};
#elif TEST_ERR == 1
  const auto count = cuda::args::constant<0, float>{};
#elif TEST_ERR == 2
  const auto count = cuda::args::deferred<float*>{nullptr};
#else
  const auto count = cuda::args::immediate<int*>{nullptr};
#endif

  // expected-error {{"NumItemsT must be an integral type, but not bool"}}
  const auto error = cub::DeviceScan::ExclusiveSum(items, items, count);
  return error == cudaSuccess ? 0 : 1;
}
