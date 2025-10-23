//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/memory>
#include <cuda/std/cassert>

#include <cuda_runtime_api.h>

#include "test_macros.h"

__device__ int device_ptr2[]              = {1, 2, 3, 4};
__device__ __managed__ int managed_ptr2[] = {1, 2, 3, 4};

int host_ptr5[] = {1, 2, 3, 4};

template <typename Pointer>
void test_accessible_pointer(Pointer ptr, bool is_host_accessible, bool is_device_accessible, bool is_managed_accessible)
{
  assert(cuda::is_host_accessible(ptr) == is_host_accessible);
  assert(cuda::is_device_accessible(ptr) == is_device_accessible);
  assert(cuda::is_managed_pointer(ptr) == is_managed_accessible);
  if constexpr (!cuda::std::is_same_v<Pointer, void*> && !cuda::std::is_same_v<Pointer, const void*>)
  {
    assert(cuda::is_host_accessible(ptr + 1) == is_host_accessible);
    assert(cuda::is_device_accessible(ptr + 1) == is_device_accessible);
    assert(cuda::is_managed_pointer(ptr + 1) == is_managed_accessible);
  }
}

bool test()
{
  int host_ptr1[] = {1, 2, 3, 4};
  auto host_ptr2  = new int[2];
  int* host_ptr3  = nullptr;
  assert(cudaMallocHost(&host_ptr3, sizeof(int) * 2) == cudaSuccess);

  int* host_ptr4 = nullptr;
  assert(cudaHostAlloc(&host_ptr4, sizeof(int) * 2, cudaHostAllocMapped) == cudaSuccess);

  int* device_ptr1 = nullptr;
  assert(cudaMalloc(&device_ptr1, sizeof(int) * 2) == cudaSuccess);

  int* managed_ptr1 = nullptr;
  assert(cudaMallocManaged(&managed_ptr1, sizeof(int) * 2) == cudaSuccess);

  test_accessible_pointer((void*) nullptr, true, true, true);

  test_accessible_pointer(host_ptr1, true, true, true); // memory space cannot be verified for local array
  test_accessible_pointer(host_ptr2, true, true, true); // memory space cannot be verified for non-cuda malloc
  test_accessible_pointer(host_ptr3, true, false, false);
  test_accessible_pointer(host_ptr4, true, false, false);
  test_accessible_pointer(host_ptr5, true, true, true); // memory space cannot be verified for global array

  test_accessible_pointer(device_ptr1, false, true, false);
  test_accessible_pointer(device_ptr2, true, true, true); // memory space cannot be verified for global device array
  void* device_ptr3 = nullptr;
  assert(cudaGetSymbolAddress(&device_ptr3, device_ptr2) == cudaSuccess);
  test_accessible_pointer(device_ptr3, false, true, false);

  const int* const_device_ptr1 = device_ptr1;
  test_accessible_pointer(const_device_ptr1, false, true, false);

  test_accessible_pointer(managed_ptr1, true, true, true);
  test_accessible_pointer(managed_ptr2, true, true, true);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test());))
  return 0;
}
