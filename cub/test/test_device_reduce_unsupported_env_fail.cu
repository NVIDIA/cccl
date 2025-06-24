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

  // expected-error {{"gpu_to_gpu determinism is only supported for integral types, or float and double types with
  // ::cuda::std::plus operator, or any floating point types with ::cuda::minimum<> or ::cuda::maximum<> operators."}}
  cub::DeviceReduce::Reduce(ptr, ptr, 0, cuda::std::plus<float>{}, cuda::std::complex<float>{}, env);
}
