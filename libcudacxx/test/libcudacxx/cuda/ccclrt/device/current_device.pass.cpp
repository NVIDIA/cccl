//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// WANTS_CUDADEVRT.

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

__device__ void test_current_device(cuda::device_ref expected)
{
  assert(cuda::device::current_device() == expected);
}

#if !_CCCL_COMPILER(NVRTC)

__global__ void test_current_device_kernel(cuda::device_ref expected)
{
  test_current_device(expected);
}

__host__ void test_each_device()
{
  for (auto device : cuda::devices)
  {
    assert(cudaSetDevice(device.get()) == cudaSuccess);
    test_current_device_kernel<<<1, 1>>>(device);
    assert(cudaDeviceSynchronize() == cudaSuccess);
  }
}

#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
#if !defined(TEST_NO_CUDADEVRT)
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test_each_device();), (test_current_device(cuda::device_ref{0});))
#endif // !TEST_NO_CUDADEVRT
  return 0;
}
