//===----------------------------------------------------------------------===//
//
// Part of CUDA C++ Core Libraries, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#define CUDART_REQUIRE(call) REQUIRE((call) == cudaSuccess)

__global__ void increment_kernel(int* p, int n)
{
  int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < n)
  {
    p[idx] += 1;
  }
}

TEST_CASE("CUDA device is available", "[cuda_smoke]")
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
