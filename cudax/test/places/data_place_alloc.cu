//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Test that data_place can be used to allocate/deallocate memory
 *        directly without a CUDASTF context.
 *
 * This demonstrates how places can be used for raw memory allocation
 * outside of the task-based programming model.
 */

#include <cuda/experimental/__places/exec/locality_domain.cuh>
#include <cuda/experimental/__places/places.cuh>

#include <cstdio>

using namespace cuda::experimental::places;

__global__ void init_kernel(int* ptr, int n, int value)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    ptr[tid] = value + tid;
  }
}

__global__ void check_kernel(int* ptr, int n, int value, int* result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    if (ptr[tid] != value + tid)
    {
      atomicExch(result, 1); // Set error flag
    }
  }
}

void test_host_allocation()
{
  printf("Testing host allocation...\n");

  const size_t n         = 1024;
  const size_t byte_size = n * sizeof(int);

  // Allocate using data_place::host() - stream parameter is ignored for host allocations
  auto place = data_place::host();
  EXPECT(!place.allocation_is_stream_ordered()); // Host allocations are blocking

  int* ptr = static_cast<int*>(place.allocate(byte_size));
  EXPECT(ptr != nullptr);

  // Initialize on host
  for (size_t i = 0; i < n; i++)
  {
    ptr[i] = static_cast<int>(i * 2);
  }

  // Verify
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(ptr[i] == static_cast<int>(i * 2));
  }

  // Deallocate
  place.deallocate(ptr, byte_size, nullptr);

  printf("  Host allocation test PASSED\n");
}

void test_device_allocation()
{
  printf("Testing device allocation...\n");

  const size_t n         = 1024;
  const size_t byte_size = n * sizeof(int);
  const int test_value   = 42;

  // Create a stream for the allocation
  cudaStream_t stream;
  cuda_try(cudaStreamCreate(&stream));

  // Allocate using data_place::device(0)
  auto place = data_place::device(0);
  EXPECT(place.allocation_is_stream_ordered()); // Device allocations are stream-ordered

  int* d_ptr = static_cast<int*>(place.allocate(byte_size, stream));
  EXPECT(d_ptr != nullptr);

  // Initialize on device
  init_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_ptr, n, test_value);

  // Allocate result flag on host for checking
  int* d_result;
  cuda_try(cudaMallocAsync(&d_result, sizeof(int), stream));
  cuda_try(cudaMemsetAsync(d_result, 0, sizeof(int), stream));

  // Check on device
  check_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_ptr, n, test_value, d_result);

  // Copy result back
  int h_result = 0;
  cuda_try(cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_try(cudaStreamSynchronize(stream));

  EXPECT(h_result == 0); // No errors

  // Cleanup
  cuda_try(cudaFreeAsync(d_result, stream));
  place.deallocate(d_ptr, byte_size, stream);

  cuda_try(cudaStreamSynchronize(stream));
  cuda_try(cudaStreamDestroy(stream));

  printf("  Device allocation test PASSED\n");
}

void test_managed_allocation()
{
  printf("Testing managed allocation...\n");

  // Check if concurrent managed access is supported
  int dev;
  cuda_try(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  cuda_try(cudaGetDeviceProperties(&prop, dev));
  if (!prop.concurrentManagedAccess)
  {
    printf("  Concurrent CPU/GPU access not supported, skipping managed test.\n");
    return;
  }

  const size_t n         = 1024;
  const size_t byte_size = n * sizeof(int);
  const int test_value   = 100;

  cudaStream_t stream;
  cuda_try(cudaStreamCreate(&stream));

  // Allocate using data_place::managed()
  auto place = data_place::managed();
  EXPECT(!place.allocation_is_stream_ordered()); // Managed allocations are immediate (stream ignored)

  int* ptr = static_cast<int*>(place.allocate(byte_size));
  EXPECT(ptr != nullptr);

  // Initialize on host (managed memory is accessible from both CPU and GPU)
  for (size_t i = 0; i < n; i++)
  {
    ptr[i] = test_value + static_cast<int>(i);
  }

  // Read back on device and verify
  int* d_result;
  cuda_try(cudaMallocAsync(&d_result, sizeof(int), stream));
  cuda_try(cudaMemsetAsync(d_result, 0, sizeof(int), stream));

  check_kernel<<<(n + 255) / 256, 256, 0, stream>>>(ptr, n, test_value, d_result);

  int h_result = 0;
  cuda_try(cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_try(cudaStreamSynchronize(stream));

  EXPECT(h_result == 0); // No errors

  // Cleanup
  cuda_try(cudaFreeAsync(d_result, stream));
  cuda_try(cudaStreamSynchronize(stream));
  place.deallocate(ptr, byte_size);

  cuda_try(cudaStreamDestroy(stream));

  printf("  Managed allocation test PASSED\n");
}

// A locality-domain data place must never change the release-threshold policy
// of the process-global device default pool. That pool is shared with every
// other `cudaMallocAsync` user in the process, so retention there is not this
// place's decision to make — most visibly on machines with no locality
// domains, where the place degrades to whole-device memory and would
// otherwise be configuring a pool it does not own. (The domain's own default
// pool is a different location and is configured by the library-wide
// accessor.)
void test_device_default_pool_policy_untouched()
{
  printf("Testing that a locality-domain place leaves the device default pool alone...\n");

  cudaMemPool_t default_pool;
  cuda_try(cudaDeviceGetDefaultMemPool(&default_pool, 0));
  cuuint64_t before = 0;
  cuda_try(cudaMemPoolGetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &before));

  cudaStream_t stream;
  cuda_try(cudaStreamCreate(&stream));

  auto place     = data_place::locality_domain(0, 0);
  const size_t n = size_t{1} << 20;
  void* ptr      = place.allocate(static_cast<::std::ptrdiff_t>(n), stream);
  cuda_try(cudaStreamSynchronize(stream));
  place.deallocate(ptr, n, stream);
  cuda_try(cudaStreamSynchronize(stream));

  cuuint64_t after = 0;
  cuda_try(cudaMemPoolGetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &after));
  EXPECT(before == after);

  cuda_try(cudaStreamDestroy(stream));
  printf("  Device default pool policy test PASSED\n");
}

int main()
{
  printf("=== Testing data_place direct allocation (no context) ===\n\n");

  test_host_allocation();
  test_device_allocation();
  test_managed_allocation();
  test_device_default_pool_policy_untouched();

  printf("\n=== All tests PASSED ===\n");
  return 0;
}
