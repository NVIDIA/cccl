//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "cuda_smoke_common.cuh"

// smoke test for pinned host memory

TEST_CASE("cudaMallocHost round-trip works", "[cuda_smoke][pinned_memory]")
{
  (void) cudaGetLastError();

  constexpr int n = 256;

  int* h_pinned = nullptr;
  CUDART_REQUIRE(cudaMallocHost(&h_pinned, n * sizeof(int)));
  REQUIRE(h_pinned != nullptr);

  int* d_ptr = nullptr;
  CUDART_REQUIRE(cudaMalloc(&d_ptr, n * sizeof(int)));
  REQUIRE(d_ptr != nullptr);

  for (int i = 0; i < n; ++i)
  {
    h_pinned[i] = i;
  }
  CUDART_REQUIRE(cudaMemcpy(d_ptr, h_pinned, n * sizeof(int), cudaMemcpyHostToDevice));

  increment_kernel<<<4, 64>>>(d_ptr, n);
  CUDART_REQUIRE(cudaGetLastError());
  CUDART_REQUIRE(cudaDeviceSynchronize());

  CUDART_REQUIRE(cudaMemcpy(h_pinned, d_ptr, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
  {
    REQUIRE(h_pinned[i] == i + 1);
  }

  CUDART_REQUIRE(cudaFree(d_ptr));
  CUDART_REQUIRE(cudaFreeHost(h_pinned));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
