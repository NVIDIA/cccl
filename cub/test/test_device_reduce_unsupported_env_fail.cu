// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_reduce.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/complex>

int main()
{
  namespace stdexec = cuda::std::execution;

  cuda::std::complex<float>* ptr{};
  auto env = cuda::execution::require(cuda::execution::determinism::gpu_to_gpu);

  // expected-error {{"gpu-to-gpu deterministic reduction supports only float and double sum."}}
  cub::DeviceReduce::Reduce(ptr, ptr, 0, cuda::std::plus<>{}, 0, env);
}
