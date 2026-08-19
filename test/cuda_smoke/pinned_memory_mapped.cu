//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_smoke_common.cuh>

// smoke test for mapped pinned host memory

TEST_CASE("cudaHostAlloc mapped (zero-copy) works", "[cuda_smoke][pinned_memory][mapped]")
{
  (void) cudaGetLastError();

  int can_map = 0;
  CUDART_REQUIRE(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0));
  if (!can_map)
  {
    SKIP("Device cannot map host memory (cudaDevAttrCanMapHostMemory == 0).");
  }

  constexpr int n = 256;

  int* h_mapped = nullptr;
  CUDART_REQUIRE(cudaHostAlloc(&h_mapped, n * sizeof(int), cudaHostAllocMapped));
  REQUIRE(h_mapped != nullptr);

  for (int i = 0; i < n; ++i)
  {
    h_mapped[i] = i;
  }

  int* d_view = nullptr;
  CUDART_REQUIRE(cudaHostGetDevicePointer(&d_view, h_mapped, 0));
  REQUIRE(d_view != nullptr);

  increment_kernel<<<4, 64>>>(d_view, n);
  CUDART_REQUIRE(cudaGetLastError());
  CUDART_REQUIRE(cudaDeviceSynchronize());

  for (int i = 0; i < n; ++i)
  {
    REQUIRE(h_mapped[i] == i + 1);
  }

  CUDART_REQUIRE(cudaFreeHost(h_mapped));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
