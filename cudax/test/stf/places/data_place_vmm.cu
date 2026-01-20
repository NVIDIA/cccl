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
 * @brief Test that data_place::mem_create() can be used to create VMM-based
 *        physical memory allocations.
 *
 * This tests the low-level VMM allocation interface used by localized arrays
 * (composite_slice) for creating physical memory segments that are mapped
 * into a contiguous virtual address space.
 */

#include <cuda/experimental/__stf/places/places.cuh>

#include <cstdio>

using namespace cuda::experimental::stf;

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

// Check if VMM is supported on the current device
bool vmm_supported(int dev_id = 0)
{
  CUdevice dev;
  cuda_safe_call(cuDeviceGet(&dev, dev_id));
  int supportsVMM;
  cuda_safe_call(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
  return supportsVMM == 1;
}

// Get allocation granularity for VMM
size_t get_granularity(int dev_id)
{
  CUmemAllocationProp prop = {};
  prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id         = dev_id;

  size_t granularity;
  cuda_safe_call(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return granularity;
}

void test_device_vmm_allocation()
{
  printf("Testing device VMM allocation (mem_create)...\n");

  int dev_id = 0;
  cuda_safe_call(cudaSetDevice(dev_id));

  // Get allocation granularity - VMM allocations must be aligned to this
  size_t granularity = get_granularity(dev_id);
  printf("  Allocation granularity: %zu bytes\n", granularity);

  // Allocate at least one granularity unit
  const size_t alloc_size = granularity;
  const size_t n          = alloc_size / sizeof(int);
  const int test_value    = 42;

  // Create physical memory using data_place::mem_create
  auto place = data_place::device(dev_id);
  CUmemGenericAllocationHandle handle;
  CUresult result = place.mem_create(&handle, alloc_size);
  EXPECT(result == CUDA_SUCCESS);

  // Reserve virtual address space
  CUdeviceptr va_ptr;
  cuda_safe_call(cuMemAddressReserve(&va_ptr, alloc_size, 0, 0, 0));

  // Map the physical allocation to the virtual address
  cuda_safe_call(cuMemMap(va_ptr, alloc_size, 0, handle, 0));

  // Set access permissions for the current device
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id     = dev_id;
  accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuda_safe_call(cuMemSetAccess(va_ptr, alloc_size, &accessDesc, 1));

  // Now we can use the memory!
  int* d_ptr = reinterpret_cast<int*>(va_ptr);

  // Create a stream for operations
  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  // Initialize on device
  init_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_ptr, n, test_value);

  // Allocate result flag for checking
  int* d_result;
  cuda_safe_call(cudaMallocAsync(&d_result, sizeof(int), stream));
  cuda_safe_call(cudaMemsetAsync(d_result, 0, sizeof(int), stream));

  // Check on device
  check_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_ptr, n, test_value, d_result);

  // Copy result back
  int h_result = 0;
  cuda_safe_call(cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));

  EXPECT(h_result == 0); // No errors

  // Cleanup
  cuda_safe_call(cudaFreeAsync(d_result, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));
  cuda_safe_call(cudaStreamDestroy(stream));

  // Unmap and release VMM resources
  cuda_safe_call(cuMemUnmap(va_ptr, alloc_size));
  cuda_safe_call(cuMemRelease(handle));
  cuda_safe_call(cuMemAddressFree(va_ptr, alloc_size));

  printf("  Device VMM allocation test PASSED\n");
}

// Host VMM requires CU_MEM_LOCATION_TYPE_HOST which is only available in CUDA 12.2+
#if _CCCL_CTK_AT_LEAST(12, 2)
void test_host_vmm_allocation()
{
  printf("Testing host VMM allocation (mem_create)...\n");

  // Host VMM allocations use CU_MEM_LOCATION_TYPE_HOST
  // First check if host VMM is supported (requires appropriate driver/hardware)

  // Get host allocation granularity
  CUmemAllocationProp prop = {};
  prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type       = CU_MEM_LOCATION_TYPE_HOST;
  prop.location.id         = 0;

  size_t granularity;
  CUresult gran_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (gran_result != CUDA_SUCCESS)
  {
    printf("  Host VMM not supported on this system, skipping.\n");
    return;
  }

  printf("  Host allocation granularity: %zu bytes\n", granularity);

  const size_t alloc_size = granularity;
  const size_t n          = alloc_size / sizeof(int);

  // Create physical memory using data_place::mem_create with host place
  auto place = data_place::host();
  CUmemGenericAllocationHandle handle;
  CUresult result = place.mem_create(&handle, alloc_size);
  if (result != CUDA_SUCCESS)
  {
    printf("  Host mem_create not supported (error %d), skipping.\n", result);
    return;
  }

  // Reserve virtual address space
  CUdeviceptr va_ptr;
  cuda_safe_call(cuMemAddressReserve(&va_ptr, alloc_size, 0, 0, 0));

  // Map the physical allocation to the virtual address
  cuda_safe_call(cuMemMap(va_ptr, alloc_size, 0, handle, 0));

  // Set access permissions for the host
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type   = CU_MEM_LOCATION_TYPE_HOST;
  accessDesc.location.id     = 0;
  accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuda_safe_call(cuMemSetAccess(va_ptr, alloc_size, &accessDesc, 1));

  // Use the memory from the host
  int* ptr = reinterpret_cast<int*>(va_ptr);

  // Initialize on host
  for (size_t i = 0; i < n; i++)
  {
    ptr[i] = static_cast<int>(i * 2);
  }

  // Verify on host
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(ptr[i] == static_cast<int>(i * 2));
  }

  // Cleanup VMM resources
  cuda_safe_call(cuMemUnmap(va_ptr, alloc_size));
  cuda_safe_call(cuMemRelease(handle));
  cuda_safe_call(cuMemAddressFree(va_ptr, alloc_size));

  printf("  Host VMM allocation test PASSED\n");
}
#endif // _CCCL_CTK_AT_LEAST(12, 2)

void test_multi_segment_vmm()
{
  printf("Testing multi-segment VMM allocation...\n");

  int dev_id = 0;
  cuda_safe_call(cudaSetDevice(dev_id));

  size_t granularity = get_granularity(dev_id);

  // Create two segments and map them contiguously
  const size_t segment_size = granularity;
  const size_t total_size   = 2 * segment_size;
  const size_t n            = total_size / sizeof(int);
  const int test_value      = 100;

  auto place = data_place::device(dev_id);

  // Create two physical allocations
  CUmemGenericAllocationHandle handle1, handle2;
  cuda_safe_call(place.mem_create(&handle1, segment_size));
  cuda_safe_call(place.mem_create(&handle2, segment_size));

  // Reserve contiguous virtual address space for both
  CUdeviceptr va_ptr;
  cuda_safe_call(cuMemAddressReserve(&va_ptr, total_size, 0, 0, 0));

  // Map both segments contiguously
  cuda_safe_call(cuMemMap(va_ptr, segment_size, 0, handle1, 0));
  cuda_safe_call(cuMemMap(va_ptr + segment_size, segment_size, 0, handle2, 0));

  // Set access for the entire range
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id     = dev_id;
  accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuda_safe_call(cuMemSetAccess(va_ptr, total_size, &accessDesc, 1));

  int* d_ptr = reinterpret_cast<int*>(va_ptr);

  cudaStream_t stream;
  cuda_safe_call(cudaStreamCreate(&stream));

  // Initialize the entire contiguous range
  init_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_ptr, n, test_value);

  // Check the entire range
  int* d_result;
  cuda_safe_call(cudaMallocAsync(&d_result, sizeof(int), stream));
  cuda_safe_call(cudaMemsetAsync(d_result, 0, sizeof(int), stream));
  check_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_ptr, n, test_value, d_result);

  int h_result = 0;
  cuda_safe_call(cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));

  EXPECT(h_result == 0);

  // Cleanup
  cuda_safe_call(cudaFreeAsync(d_result, stream));
  cuda_safe_call(cudaStreamSynchronize(stream));
  cuda_safe_call(cudaStreamDestroy(stream));

  cuda_safe_call(cuMemUnmap(va_ptr, segment_size));
  cuda_safe_call(cuMemUnmap(va_ptr + segment_size, segment_size));
  cuda_safe_call(cuMemRelease(handle1));
  cuda_safe_call(cuMemRelease(handle2));
  cuda_safe_call(cuMemAddressFree(va_ptr, total_size));

  printf("  Multi-segment VMM allocation test PASSED\n");
}

int main()
{
  printf("=== Testing data_place VMM allocation (mem_create) ===\n\n");

  // Initialize CUDA driver API
  cuda_safe_call(cuInit(0));

  // Check VMM support
  if (!vmm_supported())
  {
    printf("VMM not supported on this device, skipping tests.\n");
    return 0;
  }

  test_device_vmm_allocation();
#if _CCCL_CTK_AT_LEAST(12, 2)
  test_host_vmm_allocation();
#endif // _CCCL_CTK_AT_LEAST(12, 2)
  test_multi_segment_vmm();

  printf("\n=== All VMM tests PASSED ===\n");
  return 0;
}
