//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// CUDA runtime smoke test
// Registered as a CTest setup fixture (cccl.cuda_runtime_ok); all other tests
// require this fixture, so a broken CUDA install fails fast with a clear cause.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#define CUDART_REQUIRE(call) REQUIRE((call) == cudaSuccess)

__global__ void increment_kernel(int* p, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    p[idx] += 1;
  }
}

// All TEST_CASEs are tagged [.smoke] so they are hidden from the default
// Catch2 listing when the binary is run by hand; they are still reachable via
// `--list-tests --tags`, CTest, or by passing the tag explicitly.

TEST_CASE("CUDA device is available", "[.smoke][cuda_smoke]")
{
  int device_count = 0;
  CUDART_REQUIRE(cudaGetDeviceCount(&device_count));
  REQUIRE(device_count > 0);

  CUDART_REQUIRE(cudaSetDevice(0));

  cudaDeviceProp props{};
  CUDART_REQUIRE(cudaGetDeviceProperties(&props, 0));
  REQUIRE(props.name[0] != '\0');

  REQUIRE(cudaGetLastError() == cudaSuccess);
}

// Covers the NVBug 5739038 failure : host write -> kernel transform -> host read-back
// via a managed pointer. Fails here instead of producing dozens of thrust/cub failures.
TEST_CASE("cudaMallocManaged round-trip works", "[.smoke][cuda_smoke][managed_memory]")
{
  (void) cudaGetLastError(); // clear any pre-existing error state

  int managed_supported = 0;
  CUDART_REQUIRE(cudaDeviceGetAttribute(&managed_supported, cudaDevAttrManagedMemory, 0));
  if (!managed_supported)
  {
    SKIP("Device does not support managed memory (cudaDevAttrManagedMemory == 0).");
  }

  constexpr int n = 256;
  int* p          = nullptr;
  CUDART_REQUIRE(cudaMallocManaged(&p, n * sizeof(int)));

  for (int i = 0; i < n; ++i) // host write
  {
    p[i] = i;
  }
  CUDART_REQUIRE(cudaDeviceSynchronize());

  constexpr int block = 64;
  const int grid      = (n + block - 1) / block;
  increment_kernel<<<grid, block>>>(p, n); // device transform
  CUDART_REQUIRE(cudaGetLastError());
  CUDART_REQUIRE(cudaDeviceSynchronize());

  for (int i = 0; i < n; ++i) // host read-back
  {
    REQUIRE(p[i] == i + 1);
  }

  CUDART_REQUIRE(cudaFree(p));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
