//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>
#include <cuda/memory>
#include <cuda/std/cassert>

#include <cuda_runtime_api.h>

#include "test_macros.h"

__device__ int device_ptr3[]              = {1, 2, 3, 4};
__device__ __managed__ int managed_ptr2[] = {1, 2, 3, 4};

int host_ptr5[] = {1, 2, 3, 4};

template <typename Pointer>
void test_accessible_pointer(
  Pointer ptr, bool is_host_accessible, bool is_device_accessible, bool is_managed_accessible, cuda::device_ref device)
{
  assert(cuda::is_host_accessible(ptr) == is_host_accessible);
  assert(cuda::is_device_accessible(ptr, device) == is_device_accessible);
  assert(cuda::is_managed(ptr) == is_managed_accessible);
  if constexpr (!cuda::std::is_same_v<Pointer, void*> && !cuda::std::is_same_v<Pointer, const void*>)
  {
    assert(cuda::is_host_accessible(ptr + 1) == is_host_accessible);
    assert(cuda::is_device_accessible(ptr + 1, device) == is_device_accessible);
    assert(cuda::is_managed(ptr + 1) == is_managed_accessible);
  }
}

bool test_basic()
{
  cuda::device_ref dev{0};
  int host_ptr1[] = {1, 2, 3, 4};
  auto host_ptr2  = new int[2];
  int* host_ptr3  = nullptr;
  assert(cudaMallocHost(&host_ptr3, sizeof(int) * 2) == cudaSuccess);

  int* host_ptr4 = nullptr;
  assert(cudaHostAlloc(&host_ptr4, sizeof(int) * 2, cudaHostAllocMapped) == cudaSuccess);

  int* device_ptr1 = nullptr;
  assert(cudaMalloc(&device_ptr1, sizeof(int) * 2) == cudaSuccess);

  int* device_ptr2    = nullptr;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  assert(cudaMallocAsync(&device_ptr2, sizeof(int) * 2, stream) == cudaSuccess);

  int* managed_ptr1 = nullptr;
  assert(cudaMallocManaged(&managed_ptr1, sizeof(int) * 2) == cudaSuccess);

  test_accessible_pointer((void*) nullptr, true, true, true, dev);

  test_accessible_pointer(host_ptr1, true, true, true, dev); // memory space cannot be verified for local array
  test_accessible_pointer(host_ptr2, true, true, true, dev); // memory space cannot be verified for non-cuda malloc
  test_accessible_pointer(host_ptr3, true, false, false, dev);
  test_accessible_pointer(host_ptr4, true, false, false, dev);
  test_accessible_pointer(host_ptr5, true, true, true, dev); // memory space cannot be verified for global array

  test_accessible_pointer(device_ptr1, false, true, false, dev);
  test_accessible_pointer(device_ptr2, false, true, false, dev);
  test_accessible_pointer(device_ptr3, true, true, true, dev); // memory space cannot be verified for global device

  void* device_ptr4 = nullptr;
  assert(cudaGetSymbolAddress(&device_ptr4, device_ptr3) == cudaSuccess);
  test_accessible_pointer(device_ptr4, false, true, false, dev);

  const int* const_device_ptr1 = device_ptr1;
  test_accessible_pointer(const_device_ptr1, false, true, false, dev);

  test_accessible_pointer(managed_ptr1, true, true, true, dev);
  test_accessible_pointer(managed_ptr2, true, true, true, dev);
  return true;
}

bool test_memory_pool()
{
  cuda::device_ref dev{0};
  cudaMemPoolProps pool_prop = {};
  pool_prop.allocType        = cudaMemAllocationTypePinned;
  pool_prop.location.id      = dev.get();
  pool_prop.location.type    = cudaMemLocationTypeDevice;
  cudaMemPool_t mem_pool     = nullptr;
  cudaMemPoolCreate(&mem_pool, &pool_prop);

  int* device_ptr2    = nullptr;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  assert(cudaMallocFromPoolAsync(&device_ptr2, sizeof(int) * 2, mem_pool, stream) == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  test_accessible_pointer(device_ptr2, false, true, false, dev);
  return true;
}

bool test_multiple_devices()
{
  if (cuda::devices.size() < 2)
  {
    return true;
  }
  cuda::device_ref dev0{0};
  cuda::device_ref dev1{1};

  /// DEVICE 0 CONTEXT
  int* device_ptr0 = nullptr;
  assert(cudaMalloc(&device_ptr0, sizeof(int) * 2) == cudaSuccess);

  /// DEVICE 1 CONTEXT
  cuda::__ensure_current_context ctx1(dev1);
  assert(cuda::is_device_accessible(device_ptr0, dev0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev1) == false);

  int can_access_peer = 0;
  assert(cudaDeviceCanAccessPeer(&can_access_peer, dev0.get(), dev1.get()) == cudaSuccess);
  if (!can_access_peer)
  {
    return true;
  }
  assert(cuda::is_device_accessible(device_ptr0, dev1) == false);

  assert(cudaDeviceEnablePeerAccess(dev1.get(), 0) == cudaSuccess);
  assert(cuda::is_device_accessible(device_ptr0, dev0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev1) == true);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_basic());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_multiple_devices());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_memory_pool());))
  return 0;
}
