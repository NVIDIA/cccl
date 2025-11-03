//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/atomic>

// Basic smoke test for cuda::atomic_ref / cuda::std::atomic_ref on narrow (<=1B) types.

#include <cuda/atomic>
#define _LIBCUDACXX_ATOMIC_REF_ENABLE_MEMCHECK_SAFE

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include <cuda_runtime.h>

using byte = signed char;

__global__ void kernel_fetch_add(byte* value)
{
  cuda::atomic_ref<byte, cuda::thread_scope_device> ref(*value);
  ref.fetch_add(1, cuda::std::memory_order_relaxed);
}

__global__ void kernel_fetch_min(byte* value)
{
  cuda::atomic_ref<byte, cuda::thread_scope_device> ref(*value);
  const byte candidate = static_cast<byte>(1);
  ref.fetch_min(candidate, cuda::std::memory_order_relaxed);
}

__global__ void kernel_fetch_max(byte* value)
{
  cuda::atomic_ref<byte, cuda::thread_scope_device> ref(*value);
  const byte candidate = static_cast<byte>(3);
  ref.fetch_max(candidate, cuda::std::memory_order_relaxed);
}

static void run_device_tests()
{
  byte* device_value = nullptr;
  cudaError_t status = cudaMallocManaged(&device_value, sizeof(byte));
  assert(status == cudaSuccess);

  *device_value = 0;
  kernel_fetch_add<<<1, 1>>>(device_value);
  status = cudaDeviceSynchronize();
  assert(status == cudaSuccess);
  assert(*device_value == 1);

  *device_value = 100;
  kernel_fetch_min<<<1, 1>>>(device_value);
  status = cudaDeviceSynchronize();
  assert(status == cudaSuccess);
  assert(*device_value == 1);

  *device_value = 0;
  kernel_fetch_max<<<1, 1>>>(device_value);
  status = cudaDeviceSynchronize();
  assert(status == cudaSuccess);
  assert(*device_value == 3);

  cudaFree(device_value);
}

static void run_host_tests()
{
  byte host_value = 5;
  cuda::std::atomic_ref<byte> host_ref(host_value);
  (void) host_ref.is_lock_free();

  host_ref.fetch_add(3, cuda::std::memory_order_relaxed);
  assert(host_value == 8);

  byte expected  = 8;
  bool exchanged = host_ref.compare_exchange_strong(expected, static_cast<byte>(-10));
  assert(exchanged);
  assert(host_value == static_cast<byte>(-10));

  expected  = static_cast<byte>(-5);
  exchanged = host_ref.compare_exchange_strong(expected, static_cast<byte>(42));
  assert(!exchanged);
  assert(expected == static_cast<byte>(-10));
}

int main(int, char**)
{
  run_host_tests();
  run_device_tests();
  return 0;
}
