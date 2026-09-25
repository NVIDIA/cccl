// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

struct invalid_object
{};

int main()
{
  int* ptr{};

  // expected-error {{"An object passed as a DeviceReduce environment must provide a stream or a memory resource."}}
  auto error = cub::DeviceReduce::Sum(ptr, ptr, 0, invalid_object{});
  if (error != cudaSuccess)
  {
    return 1;
  }
}
