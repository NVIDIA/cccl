//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_smoke_common.cuh>

TEST_CASE("cudaMallocManaged round-trip works", "[cuda_smoke][managed_memory]")
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

  increment_kernel<<<4, 64>>>(p, n); // device transform
  CUDART_REQUIRE(cudaGetLastError());
  CUDART_REQUIRE(cudaDeviceSynchronize());

  for (int i = 0; i < n; ++i) // host read-back
  {
    REQUIRE(p[i] == i + 1);
  }

  CUDART_REQUIRE(cudaFree(p));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
