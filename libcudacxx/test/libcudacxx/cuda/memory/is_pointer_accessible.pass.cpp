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

__device__ int device_ptr1[]              = {1, 2, 3, 4};
__device__ __managed__ int managed_ptr1[] = {1, 2, 3, 4};

int host_ptr1[] = {1, 2, 3, 4};

template <typename Pointer>
void test_accessible_pointer(
  Pointer ptr, bool is_host_accessible, bool is_device_accessible, bool is_managed_accessible, cuda::device_ref device)
{
  assert(cuda::is_host_accessible(ptr) == is_host_accessible);
  assert(cuda::is_device_accessible(ptr, device) == is_device_accessible);
  assert(cuda::__is_device_or_managed_memory(ptr) == is_device_accessible);
  assert(cuda::is_managed(ptr) == is_managed_accessible);
  if constexpr (!cuda::std::is_same_v<Pointer, const void*> && !cuda::std::is_same_v<Pointer, void*>)
  {
    assert(cuda::is_host_accessible(ptr + 1) == is_host_accessible);
    assert(cuda::is_device_accessible(ptr + 1, device) == is_device_accessible);
    assert(cuda::__is_device_or_managed_memory(ptr + 1) == is_device_accessible);
    assert(cuda::is_managed(ptr + 1) == is_managed_accessible);
  }
}

bool test_basic()
{
  cuda::device_ref dev{0};
  [[maybe_unused]] int host_ptr2[] = {1, 2, 3, 4};
  [[maybe_unused]] auto host_ptr3  = new int[2];
  [[maybe_unused]] int* host_ptr4  = nullptr;
  assert(cudaMallocHost(&host_ptr4, sizeof(int) * 2) == cudaSuccess);

  int* host_ptr5 = nullptr;
  assert(cudaHostAlloc(&host_ptr5, sizeof(int) * 2, cudaHostAllocMapped) == cudaSuccess);

  int* device_ptr2 = nullptr;
  assert(cudaMalloc(&device_ptr2, sizeof(int) * 2) == cudaSuccess);

  int* device_ptr3    = nullptr;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  assert(cudaMallocAsync(&device_ptr3, sizeof(int) * 2, stream) == cudaSuccess);

  int* managed_ptr2 = nullptr;
  assert(cudaMallocManaged(&managed_ptr2, sizeof(int) * 2) == cudaSuccess);

  test_accessible_pointer((void*) nullptr, false, false, false, dev);

  test_accessible_pointer(host_ptr1, true, false, false, dev); // global host array
  test_accessible_pointer(host_ptr2, true, false, false, dev); // local host array
  test_accessible_pointer(host_ptr3, true, false, false, dev); // non-cuda malloc host memory
  test_accessible_pointer(host_ptr4, true, false, false, dev); // stack-allocated host memory
  test_accessible_pointer(host_ptr5, true, false, false, dev); // pinned host memory

  test_accessible_pointer(device_ptr2, false, true, false, dev); // cudaMalloc device pointer
  test_accessible_pointer(device_ptr3, false, true, false, dev); // cudaMallocAsync device pointer

  void* device_ptr4 = nullptr;
  assert(cudaGetSymbolAddress(&device_ptr4, device_ptr1) == cudaSuccess);
  test_accessible_pointer(device_ptr4, false, true, false, dev); // cudaGetSymbolAddress device pointer

  const int* const_device_ptr2 = device_ptr2;
  test_accessible_pointer(const_device_ptr2, false, true, false, dev); // const device pointer

  test_accessible_pointer(managed_ptr1, true, true, true, dev); // global managed memory
  test_accessible_pointer(managed_ptr2, true, true, true, dev); // allocated managed memory
  return true;
}

void* allocate_memory_from_pool(
  cudaMemAllocationType alloc_type, cudaMemLocationType location_type, cuda::device_ref dev)
{
  cudaMemPoolProps pool_prop = {};
  pool_prop.allocType        = alloc_type;
  pool_prop.location.id      = dev.get();
  pool_prop.location.type    = location_type;
  cudaMemPool_t mem_pool     = nullptr;
  assert(cudaMemPoolCreate(&mem_pool, &pool_prop) == cudaSuccess);

  int* ptr            = nullptr;
  cudaStream_t stream = nullptr;
  assert(cudaStreamCreate(&stream) == cudaSuccess);
  assert(cudaMallocFromPoolAsync(&ptr, sizeof(int) * 2, mem_pool, stream) == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudaMemAccessDesc access_desc = {};
  access_desc.flags             = cudaMemAccessFlagsProtReadWrite;
  access_desc.location.type     = location_type;
  access_desc.location.id       = dev.get();
  assert(cudaMemPoolSetAccess(mem_pool, &access_desc, 1) == cudaSuccess);
  return ptr;
}

void test_memory_pool_impl(
  cudaMemAllocationType alloc_type,
  cudaMemLocationType location_type,
  bool is_host_accessible,
  bool is_device_accessible,
  bool is_managed_accessible)
{
  cuda::device_ref dev{0};
  void* ptr = allocate_memory_from_pool(alloc_type, location_type, dev);

  test_accessible_pointer(ptr, is_host_accessible, is_device_accessible, is_managed_accessible, dev);
}

bool test_memory_pool()
{
  test_memory_pool_impl(cudaMemAllocationTypePinned, cudaMemLocationTypeDevice, false, true, false);

#if _CCCL_CTK_AT_LEAST(12, 2)
  test_memory_pool_impl(cudaMemAllocationTypePinned, cudaMemLocationTypeHost, true, false, false);
#endif // _CCCL_CTK_AT_LEAST(12, 2)
#if _CCCL_CTK_AT_LEAST(13, 0)
  // TODO(fbusato): check if this can be improved in future releases
  test_memory_pool_impl(cudaMemAllocationTypeManaged, cudaMemLocationTypeHost, true, false, true);
  test_memory_pool_impl(cudaMemAllocationTypeManaged, cudaMemLocationTypeDevice, false, true, true);
#endif // _CCCL_CTK_AT_LEAST(13, 0)
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
  assert(cuda::__is_device_or_managed_memory(device_ptr0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev1) == false);

  int can_access_peer = 0;
  assert(cudaDeviceCanAccessPeer(&can_access_peer, dev0.get(), dev1.get()) == cudaSuccess);
  if (!can_access_peer)
  {
    return true;
  }
  assert(cuda::__is_device_or_managed_memory(device_ptr0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev1) == false);

  assert(cudaDeviceEnablePeerAccess(dev1.get(), 0) == cudaSuccess);
  assert(cuda::is_device_accessible(device_ptr0, dev0) == true);
  assert(cuda::is_device_accessible(device_ptr0, dev1) == true);
  return true;
}

bool test_multiple_devices_from_pool()
{
  if (cuda::devices.size() < 2)
  {
    return true;
  }
  cuda::device_ref dev0{0};
  cuda::device_ref dev1{1};

  void* ptr = allocate_memory_from_pool(cudaMemAllocationTypePinned, cudaMemLocationTypeDevice, dev0);

  /// DEVICE 1 CONTEXT
  cuda::__ensure_current_context ctx1(dev1);
  int can_access_peer = 0;
  assert(cudaDeviceCanAccessPeer(&can_access_peer, dev0.get(), dev1.get()) == cudaSuccess);
  if (!can_access_peer)
  {
    return true;
  }
  assert(cuda::is_device_accessible(ptr, dev1) == false);
  assert(cuda::__is_device_or_managed_memory(ptr) == true);

  assert(cudaDeviceEnablePeerAccess(dev1.get(), 0) == cudaSuccess);
  assert(cuda::is_device_accessible(ptr, dev0) == true);
  assert(cuda::is_device_accessible(ptr, dev1) == true);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_basic());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_memory_pool());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_multiple_devices());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_multiple_devices_from_pool());))
  return 0;
}
