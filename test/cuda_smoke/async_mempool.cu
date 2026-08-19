//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_smoke_common.cuh>

// smoke test for async memory allocations from the default memory pool

TEST_CASE("cudaMallocAsync from default pool works", "[cuda_smoke][async_mempool]")
{
  (void) cudaGetLastError();

  int pools_supported = 0;
  CUDART_REQUIRE(cudaDeviceGetAttribute(&pools_supported, cudaDevAttrMemoryPoolsSupported, 0));
  if (!pools_supported)
  {
    SKIP("Device does not support memory pools (cudaDevAttrMemoryPoolsSupported == 0).");
  }

  cudaMemPool_t pool{};
  CUDART_REQUIRE(cudaDeviceGetDefaultMemPool(&pool, 0));
  REQUIRE(pool != nullptr);

  cudaStream_t stream{};
  CUDART_REQUIRE(cudaStreamCreate(&stream));

  constexpr int n = 256;

  int* d_ptr = nullptr;
  CUDART_REQUIRE(cudaMallocFromPoolAsync(&d_ptr, n * sizeof(int), pool, stream));
  REQUIRE(d_ptr != nullptr);

  int h_ins[n];
  for (int i = 0; i < n; ++i)
  {
    h_ins[i] = i;
  }
  CUDART_REQUIRE(cudaMemcpyAsync(d_ptr, h_ins, n * sizeof(int), cudaMemcpyHostToDevice, stream));

  increment_kernel<<<4, 64, 0, stream>>>(d_ptr, n);
  CUDART_REQUIRE(cudaGetLastError());

  int h_outs[n];
  CUDART_REQUIRE(cudaMemcpyAsync(h_outs, d_ptr, n * sizeof(int), cudaMemcpyDeviceToHost, stream));

  // Stream-ordered free (deferred until stream reaches this point)
  CUDART_REQUIRE(cudaFreeAsync(d_ptr, stream));

  CUDART_REQUIRE(cudaStreamSynchronize(stream));

  for (int i = 0; i < n; ++i)
  {
    REQUIRE(h_outs[i] == i + 1);
  }

  CUDART_REQUIRE(cudaStreamDestroy(stream));
  REQUIRE(cudaGetLastError() == cudaSuccess);
}
