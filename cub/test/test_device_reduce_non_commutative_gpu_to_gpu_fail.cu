// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_reduce.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>

#include <iostream>

int main()
{
  namespace stdexec = cuda::std::execution;

  int* ptr{};
  auto env = stdexec::env{cuda::execution::require(cuda::execution::determinism::gpu_to_gpu)};

  // expected-error {{"gpu_to_gpu determinism is not supported"}}
  auto error = cub::DeviceReduce::ReduceNonCommutative(ptr, ptr, 0, cuda::std::plus<>{}, 0, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceNonCommutative failed with status: " << error << '\n';
  }
}
