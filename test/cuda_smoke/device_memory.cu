//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_smoke_common.cuh>

// smoke test for GPU memory allocation/deallocation

TEST_CASE("cudaMalloc/cudaFree round-trip works", "[cuda_smoke][device_memory]")
{
  (void) cudaGetLastError();

  constexpr int n = 256;

  int* d_ptr = nullptr;
  CUDART_REQUIRE(cudaMalloc(&d_ptr, n * sizeof(int)));
  REQUIRE(d_ptr != nullptr);

  int h_ins[n];
  for (int i = 0; i < n; ++i)
  {
    h_ins[i] = i;
  }
  CUDART_REQUIRE(cudaMemcpy(d_ptr, h_ins, n * sizeof(int), cudaMemcpyHostToDevice));

  increment_kernel<<<4, 64>>>(d_ptr, n);
  CUDART_REQUIRE(cudaGetLastError());
  CUDART_REQUIRE(cudaDeviceSynchronize());

  int h_outs[n];
  CUDART_REQUIRE(cudaMemcpy(h_outs, d_ptr, n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
  {
    REQUIRE(h_outs[i] == i + 1);
  }

  CUDART_REQUIRE(cudaFree(d_ptr));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
