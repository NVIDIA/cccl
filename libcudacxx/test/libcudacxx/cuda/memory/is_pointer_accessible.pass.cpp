//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: nvrtc

#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>
#include <cuda/memory>
#include <cuda/memory_pool>
#include <cuda/std/cassert>
#include <cuda/std/cstdlib>
#include <cuda/stream>

#include "test_macros.h"

using T                            = int;
constexpr cuda::std::size_t N      = 2;
constexpr cuda::std::size_t nbytes = sizeof(T) * N;

T global_host[N];
TEST_GLOBAL_VARIABLE T global_device[N];
__managed__ T global_managed[N];

void test_accessible_pointer(
  const T* ptr, bool host_accessible, bool device_accessible, bool is_managed, cuda::device_ref device)
{
  assert(cuda::is_host_accessible(ptr) == host_accessible);
  assert(cuda::__is_host_accessible_nothrow(ptr) == host_accessible);
  assert(cuda::is_device_accessible(ptr, device) == device_accessible);
  // assert(cuda::__is_device_accessible_nothrow(ptr, device) == device_accessible);
  assert(cuda::is_managed(ptr) == is_managed);
  assert(cuda::__is_managed_nothrow(ptr) == is_managed);

  // Skip ptr + 1 tests for nullptr.
  if (ptr != nullptr)
  {
    const T* ptr_next = ptr + 1;
    assert(cuda::is_host_accessible(ptr_next) == host_accessible);
    assert(cuda::__is_host_accessible_nothrow(ptr_next) == host_accessible);
    assert(cuda::is_device_accessible(ptr_next, device) == device_accessible);
    // assert(cuda::__is_device_accessible_nothrow(ptr_next, device) == device_accessible);
    assert(cuda::is_managed(ptr_next) == is_managed);
    assert(cuda::__is_managed_nothrow(ptr_next) == is_managed);
  }
}

void test_device_or_managed_memory(const T* ptr, bool is_device_or_managed_memory)
{
  assert(cuda::__is_device_or_managed_memory(ptr) == is_device_or_managed_memory);

  // Skip ptr + 1 tests for nullptr.
  if (ptr != nullptr)
  {
    assert(cuda::__is_device_or_managed_memory(ptr + 1) == is_device_or_managed_memory);
  }
}

TEST_GLOBAL_VARIABLE T* kernel_malloced_buffer;

__global__ void malloc_kernel()
{
  kernel_malloced_buffer = reinterpret_cast<T*>(cuda::std::malloc(nbytes));
}

__global__ void free_kernel()
{
  cuda::std::free(kernel_malloced_buffer);
}

void test_basic()
{
  cuda::device_ref dev{0};
  const auto always_ua    = cuda::device_attributes::unified_addressing(dev);
  const auto has_mempools = cuda::device_attributes::memory_pools_supported(dev);

  cuda::stream stream{dev};

  cuda::__ensure_current_context{dev};

  // Test nullptr is not accessible from anywhere.
  {
    T* buffer = nullptr;
    test_accessible_pointer(buffer, false, false, false, dev);
    test_device_or_managed_memory(buffer, false);
  }

  // Test global host buffer is accessible only from host.
  {
    T* buffer = global_host;
    test_accessible_pointer(global_host, true, false, false, dev);
    test_device_or_managed_memory(buffer, false);
  }

  // Test local host buffer is accessible only from host.
  {
    T buffer[N];
    test_accessible_pointer(buffer, true, false, false, dev);
    test_device_or_managed_memory(buffer, false);
  }

  // Test heap-allocated buffer is accessible only from host.
  {
    T* buffer = new T[N];
    test_accessible_pointer(buffer, true, false, false, dev);
    test_device_or_managed_memory(buffer, false);
    delete[] buffer;
  }

  // Test heap-allocated and default-registered buffer is accessible from host and from device when unified addressing
  // is allowed.
  {
    T* buffer = new T[N];
    assert(cudaHostRegister(buffer, nbytes, cudaHostRegisterDefault) == cudaSuccess);
    test_accessible_pointer(buffer, true, always_ua, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaHostUnregister(buffer) == cudaSuccess);
    delete[] buffer;
  }

  // Test heap-allocated and portable-registered buffer is accessible from host and from device when unified addressing
  // is allowed.
  {
    T* buffer = new T[N];
    assert(cudaHostRegister(buffer, nbytes, cudaHostRegisterPortable) == cudaSuccess);
    test_accessible_pointer(buffer, true, always_ua, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaHostUnregister(buffer) == cudaSuccess);
    delete[] buffer;
  }

  // Test heap-allocated and mapped-registered buffer is accessible both from host and device.
  {
    T* buffer = new T[N];
    assert(cudaHostRegister(buffer, nbytes, cudaHostRegisterMapped) == cudaSuccess);
    test_accessible_pointer(buffer, true, true, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaHostUnregister(buffer) == cudaSuccess);
    delete[] buffer;
  }

  // Test heap-allocated and portable and mapped-registered buffer is accessible both from host and device.
  {
    T* buffer = new T[N];
    assert(cudaHostRegister(buffer, nbytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess);
    test_accessible_pointer(buffer, true, true, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaHostUnregister(buffer) == cudaSuccess);
    delete[] buffer;
  }

  // Test default-allocated pinned buffer is accessible from host and from device when unified addressing is allowed.
  {
    T* buffer;
    assert(cudaHostAlloc(&buffer, nbytes, cudaHostAllocDefault) == cudaSuccess);
    test_accessible_pointer(buffer, true, always_ua, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaFreeHost(buffer) == cudaSuccess);
  }

  // Test portable-allocated pinned buffer is accessible  from host and from device when unified addressing is allowed.
  {
    T* buffer;
    assert(cudaHostAlloc(&buffer, nbytes, cudaHostAllocPortable) == cudaSuccess);
    test_accessible_pointer(buffer, true, always_ua, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaFreeHost(buffer) == cudaSuccess);
  }

  // Test mapped-allocated pinned buffer is accessible from both host and device.
  {
    T* buffer;
    assert(cudaHostAlloc(&buffer, nbytes, cudaHostAllocMapped) == cudaSuccess);
    test_accessible_pointer(buffer, true, true, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaFreeHost(buffer) == cudaSuccess);
  }

  // Test portable and mapped-allocated pinned buffer is accessible from both host and device.
  {
    T* buffer;
    assert(cudaHostAlloc(&buffer, nbytes, cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess);
    test_accessible_pointer(buffer, true, true, false, dev);
    test_device_or_managed_memory(buffer, false);
    assert(cudaFreeHost(buffer) == cudaSuccess);
  }

  // Test device global buffer is accessible only from device.
  {
    T* buffer;
    assert(cudaGetSymbolAddress((void**) &buffer, global_device) == cudaSuccess);
    test_accessible_pointer(buffer, false, true, false, dev);
    test_device_or_managed_memory(buffer, true);
  }

  // Test device global-allocated buffer is accessible only from device.
  {
    T* buffer;
    assert(cudaMalloc(&buffer, nbytes) == cudaSuccess);
    test_accessible_pointer(buffer, false, true, false, dev);
    test_device_or_managed_memory(buffer, true);
    assert(cudaFree(buffer) == cudaSuccess);
  }

  // Test device asynchronously and global-allocated buffer from the default pool is accessible only from device.
  if (has_mempools)
  {
    T* buffer;
    assert(cudaMallocAsync(&buffer, nbytes, stream.get()) == cudaSuccess);
    test_accessible_pointer(buffer, false, true, false, dev);
    test_device_or_managed_memory(buffer, true);
    assert(cudaFreeAsync(buffer, stream.get()) == cudaSuccess);
    stream.sync();
  }

  // Test device kernel-allocated buffer is accessible only from device.
  {
    // todo(dabayer): This seems to be failing as host-accessible and not device-accessible. Uncomment/remove once nvbug
    //                6821064 is resolved.
    // malloc_kernel<<<1, 1, 0, stream.get()>>>();
    // stream.sync();

    // T* buffer;
    // assert(cudaMemcpyFromSymbol(&buffer, kernel_malloced_buffer, nbytes) == cudaSuccess);
    // test_accessible_pointer(buffer, true, false, false, dev);
    // test_device_or_managed_memory(buffer, false);

    // free_kernel<<<1, 1, 0, stream.get()>>>();
    // stream.sync();
  }

  // Test global managed buffer is accessible both host and device.
  {
    T* buffer = global_managed;
    test_accessible_pointer(buffer, true, true, true, dev);
    test_device_or_managed_memory(buffer, true);
  }

  // Test allocated managed buffer is accessible both host and device.
  {
    T* buffer;
    assert(cudaMallocManaged(&buffer, nbytes) == cudaSuccess);
    test_accessible_pointer(buffer, true, true, true, dev);
    test_device_or_managed_memory(buffer, true);
    assert(cudaFree(buffer) == cudaSuccess);
  }
}

void test_memory_pool()
{
  cuda::device_ref dev{0};
  cuda::stream stream{dev};

  if (!cuda::device_attributes::memory_pools_supported(dev))
  {
    return;
  }

  // Test device-pinned memory pool.
  {
    cudaMemPoolProps mem_pool_props{};
    mem_pool_props.allocType     = cudaMemAllocationTypePinned;
    mem_pool_props.location.type = cudaMemLocationTypeDevice;
    mem_pool_props.location.id   = dev.get();

    cudaMemPool_t mem_pool;
    assert(cudaMemPoolCreate(&mem_pool, &mem_pool_props) == cudaSuccess);

    // Test pinned device buffer is accessible only from device.
    T* buffer;
    assert(cudaMallocFromPoolAsync(&buffer, nbytes, mem_pool, stream.get()) == cudaSuccess);
    test_accessible_pointer(buffer, false, true, false, dev);
    test_device_or_managed_memory(buffer, true);
    assert(cudaFreeAsync(buffer, stream.get()) == cudaSuccess);
    stream.sync();
  }

  // Test host-pinned memory pool.
#if _CCCL_CTK_AT_LEAST(12, 2)
  if (cuda::__is_host_memory_pool_supported())
  {
    cudaMemPoolProps mem_pool_props{};
    mem_pool_props.allocType     = cudaMemAllocationTypePinned;
    mem_pool_props.location.type = cudaMemLocationTypeHost;
    // We need to set the pool size manually on Windows due to nvbug 6816728. Remove once it's resolved.
#  if _CCCL_OS(WINDOWS)
    mem_pool_props.maxSize = 4096;
#  endif // _CCCL_OS(WINDOWS)

    cudaMemPool_t mem_pool;
    assert(cudaMemPoolCreate(&mem_pool, &mem_pool_props) == cudaSuccess);

    T* buffer;
    assert(cudaMallocFromPoolAsync(&buffer, nbytes, mem_pool, stream.get()) == cudaSuccess);

    // Test the default-allocated host buffer is inaccessible from device.
    test_accessible_pointer(buffer, true, false, false, dev);
    test_device_or_managed_memory(buffer, false);

    // Enable the access from device.
    cudaMemAccessDesc mem_access_desc{};
    mem_access_desc.location.type = cudaMemLocationTypeDevice;
    mem_access_desc.location.id   = dev.get();
    mem_access_desc.flags         = cudaMemAccessFlagsProtReadWrite;
    assert(cudaMemPoolSetAccess(mem_pool, &mem_access_desc, 1) == cudaSuccess);

    // Test the host buffer is accessible from device.
    test_accessible_pointer(buffer, true, true, false, dev);
    test_device_or_managed_memory(buffer, true);

    assert(cudaFreeAsync(buffer, stream.get()) == cudaSuccess);
    stream.sync();
  }
#endif // _CCCL_CTK_AT_LEAST(12, 2)

  // Test managed memory pool with host initial location.
#if _CCCL_CTK_AT_LEAST(13, 0)
  if (cuda::device_attributes::concurrent_managed_access(dev))
  {
    cudaMemPoolProps mem_pool_props{};
    mem_pool_props.allocType     = cudaMemAllocationTypeManaged;
    mem_pool_props.location.type = cudaMemLocationTypeHost;

    cudaMemPool_t mem_pool;
    assert(cudaMemPoolCreate(&mem_pool, &mem_pool_props) == cudaSuccess);

    T* buffer;
    assert(cudaMallocFromPoolAsync(&buffer, nbytes, mem_pool, stream.get()) == cudaSuccess);

    // Test the default-allocated managed buffer is accessible from both host and device.
    test_accessible_pointer(buffer, true, true, true, dev);
    test_device_or_managed_memory(buffer, true);

    assert(cudaFreeAsync(buffer, stream.get()) == cudaSuccess);
    stream.sync();
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  // Test managed memory pool with device initial location.
#if _CCCL_CTK_AT_LEAST(13, 0)
  if (cuda::device_attributes::concurrent_managed_access(dev))
  {
    cudaMemPoolProps mem_pool_props{};
    mem_pool_props.allocType     = cudaMemAllocationTypeManaged;
    mem_pool_props.location.type = cudaMemLocationTypeDevice;
    mem_pool_props.location.id   = dev.get();

    cudaMemPool_t mem_pool;
    assert(cudaMemPoolCreate(&mem_pool, &mem_pool_props) == cudaSuccess);

    T* buffer;
    assert(cudaMallocFromPoolAsync(&buffer, nbytes, mem_pool, stream.get()) == cudaSuccess);

    // Test the default-allocated managed buffer is accessible from both host and device.
    test_accessible_pointer(buffer, true, true, true, dev);
    test_device_or_managed_memory(buffer, true);

    assert(cudaFreeAsync(buffer, stream.get()) == cudaSuccess);
    stream.sync();
  }
#endif // _CCCL_CTK_AT_LEAST(13, 0)
}

void test_multiple_devices()
{
  cuda::device_ref dev0{0};
  cuda::device_ref dev1{1};

  const auto can_access = dev1.has_peer_access_to(dev0);

  assert(cudaSetDevice(dev0.get()) == cudaSuccess);

  T* dev0_buffer1;
  T* dev0_buffer2;
  {
    assert(cudaGetSymbolAddress((void**) &dev0_buffer1, global_device) == cudaSuccess);
    test_accessible_pointer(dev0_buffer1, false, true, false, dev0);
    test_device_or_managed_memory(dev0_buffer1, true);

    assert(cudaMalloc(&dev0_buffer2, nbytes) == cudaSuccess);
    test_accessible_pointer(dev0_buffer2, false, true, false, dev0);
    test_device_or_managed_memory(dev0_buffer2, true);
  }

  // Test that global allocated buffer on device 0 is by default not accessible on device 1.
  {
    assert(cudaSetDevice(dev1.get()) == cudaSuccess);

    test_accessible_pointer(dev0_buffer1, false, false, false, dev1);
    test_device_or_managed_memory(dev0_buffer1, true);

    test_accessible_pointer(dev0_buffer2, false, false, false, dev1);
    test_device_or_managed_memory(dev0_buffer2, true);

    assert(cudaSetDevice(dev0.get()) == cudaSuccess);
  }

  // Test that global allocated buffer on device 0 is accessible on device 1 when peer access is enabled.
  if (can_access)
  {
    assert(cudaSetDevice(dev1.get()) == cudaSuccess);
    assert(cudaDeviceEnablePeerAccess(dev0.get(), 0) == cudaSuccess);

    // todo(dabayer): It seems that __device__ variables can't be accessed even after the peer access has been enabled.
    //                Uncomment/remove once nvbug 6820994 is resolved.
    // test_accessible_pointer(dev0_buffer1, false, true, false, dev1);
    // test_device_or_managed_memory(dev0_buffer1, true);

    test_accessible_pointer(dev0_buffer2, false, true, false, dev1);
    test_device_or_managed_memory(dev0_buffer2, true);

    assert(cudaDeviceDisablePeerAccess(dev0.get()) == cudaSuccess);
    assert(cudaSetDevice(dev0.get()) == cudaSuccess);
  }

  assert(cudaFree(dev0_buffer2) == cudaSuccess);
}

void test_multiple_devices_from_pool()
{
  cuda::device_ref dev0{0};
  cuda::device_ref dev1{1};

  const auto can_access = dev1.has_peer_access_to(dev0);

  cuda::stream stream{dev0};

  if (!cuda::device_attributes::memory_pools_supported(dev0))
  {
    return;
  }

  assert(cudaSetDevice(dev0.get()) == cudaSuccess);

  // Test device-pinned memory pool.
  cudaMemPoolProps mem_pool_props{};
  mem_pool_props.allocType     = cudaMemAllocationTypePinned;
  mem_pool_props.location.type = cudaMemLocationTypeDevice;
  mem_pool_props.location.id   = dev0.get();

  cudaMemPool_t mem_pool;
  assert(cudaMemPoolCreate(&mem_pool, &mem_pool_props) == cudaSuccess);

  // Test pinned device 0 buffer is accessible only from device 0.
  T* dev0_buffer;
  assert(cudaMallocFromPoolAsync(&dev0_buffer, nbytes, mem_pool, stream.get()) == cudaSuccess);

  test_accessible_pointer(dev0_buffer, false, true, false, dev0);
  test_accessible_pointer(dev0_buffer, false, false, false, dev1);
  test_device_or_managed_memory(dev0_buffer, true);

  if (can_access)
  {
    // Enable the access from device 1.
    cudaMemAccessDesc mem_access_desc{};
    mem_access_desc.location.type = cudaMemLocationTypeDevice;
    mem_access_desc.location.id   = dev1.get();
    mem_access_desc.flags         = cudaMemAccessFlagsProtReadWrite;
    assert(cudaMemPoolSetAccess(mem_pool, &mem_access_desc, 1) == cudaSuccess);

    // Test pinned device 0 buffer is now accessible from device 1.
    test_accessible_pointer(dev0_buffer, false, true, false, dev0);
    test_accessible_pointer(dev0_buffer, false, true, false, dev1);
    test_device_or_managed_memory(dev0_buffer, true);
  }

  assert(cudaFreeAsync(dev0_buffer, stream.get()) == cudaSuccess);
  stream.sync();
}

void test()
{
  test_basic();

  if (cuda::devices.size() >= 2)
  {
    test_multiple_devices();
  }

  test_memory_pool();

  if (cuda::devices.size() >= 2)
  {
    test_multiple_devices_from_pool();
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
