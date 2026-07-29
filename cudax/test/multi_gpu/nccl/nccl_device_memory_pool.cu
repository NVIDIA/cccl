//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__device/attributes.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/nccl_device_memory_pool.h>
#include <cuda/experimental/stream.cuh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <c2h/catch2_test_helper.h>

namespace cudax = ::cuda::experimental;

namespace
{
// driver_api.h undefs the driver loading macros at the end of the file, so need to roll our own
template <class Fn>
[[nodiscard]] Fn* driver_entry_point(const char* name)
{
  auto* fn = reinterpret_cast<Fn*>(cuda::__driver::__get_driver_entry_point(name));

  REQUIRE(fn != nullptr);
  return fn;
}

// The pool is created with `cudaMemHandleTypePosixFileDescriptor`, which the driver rejects
// outright on platforms that cannot export a pool through a file descriptor. Mirrors the gating
// in libcudacxx/test/.../memory_pools.cu.
[[nodiscard]] bool default_nccl_pool_supported(cuda::device_ref device)
{
  if (!cuda::device_attributes::memory_pools_supported(device))
  {
    return false;
  }

  const auto types = cuda::device_attributes::memory_pool_supported_handle_types(device);

  return (static_cast<int>(types) & static_cast<int>(::cudaMemHandleTypePosixFileDescriptor)) != 0;
}

// The CUmem recommended allocation granularity for `device`. NCCL requires user buffers to be
// aligned to, and a multiple of, this value.
[[nodiscard]] std::size_t recommended_granularity(cuda::device_ref device)
{
  ::CUmemAllocationProp prop{};

  prop.type          = ::CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id   = device.get();

  std::size_t granularity = 0;

  static auto* fn = driver_entry_point<decltype(::cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");

  REQUIRE(fn(&granularity, &prop, ::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == ::CUDA_SUCCESS);
  REQUIRE(granularity != 0);

  return granularity;
}

// Asserts that `ptr` is device memory resident on `device`.
void require_device_resident(const void* ptr, cuda::device_ref device)
{
  REQUIRE(cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(ptr) == ::CU_MEMORYTYPE_DEVICE);
  REQUIRE(cuda::__driver::__pointerGetAttribute<::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(ptr) == device.get());
}

[[nodiscard]] cuda::device_ref require_supported_device()
{
  if (cuda::devices.size() == 0)
  {
    // Arguably we can error here
    SKIP("No CUDA devices visible");
  }

  for (auto device : cuda::devices)
  {
    if (default_nccl_pool_supported(device))
    {
      return device;
    }
  }

  SKIP("No device supports a POSIX-file-descriptor-exportable memory pool");

  return cuda::devices[0];
}
} // namespace

C2H_TEST("device_default_nccl_memory_pool interface", "[multi_gpu][nccl]")
{
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(cudax::device_default_nccl_memory_pool(cuda::std::declval<cuda::device_ref>())),
                         cuda::device_memory_pool_ref&>);

  STATIC_REQUIRE(cuda::std::is_invocable_v<decltype(cudax::device_default_nccl_memory_pool), int>);

  SECTION("satisfies the memory resource concepts of its return type")
  {
    STATIC_REQUIRE(cuda::mr::synchronous_resource_with<cuda::device_memory_pool_ref, cuda::mr::device_accessible>);
    STATIC_REQUIRE(cuda::std::is_copy_constructible_v<cuda::device_memory_pool_ref>);
  }
}

C2H_TEST("device_default_nccl_memory_pool returns a valid pool", "[multi_gpu][nccl]")
{
  const auto device = require_supported_device();

  auto& pool = cudax::device_default_nccl_memory_pool(device);

  SECTION("has a non-null native handle")
  {
    REQUIRE(pool.get() != nullptr);
  }

  SECTION("is accessible from its own device")
  {
    REQUIRE(pool.is_accessible_from(device));
  }
}

C2H_TEST("device_default_nccl_memory_pool caches per device", "[multi_gpu][nccl]")
{
  const auto device = require_supported_device();

  SECTION("repeated lookups yield the same pool")
  {
    auto& first  = cudax::device_default_nccl_memory_pool(device);
    auto& second = cudax::device_default_nccl_memory_pool(device);

    // Same object identity, not merely two refs to the same underlying `cudaMemPool_t`.
    REQUIRE(&first == &second);
    REQUIRE(first.get() == second.get());
    REQUIRE(first == second);
  }

  SECTION("an ordinal and a device_ref select the same pool")
  {
    auto& from_ref     = cudax::device_default_nccl_memory_pool(device);
    auto& from_ordinal = cudax::device_default_nccl_memory_pool(device.get());

    REQUIRE(&from_ref == &from_ordinal);
  }

  SECTION("distinct devices yield distinct pools")
  {
    for (auto other : ::cuda::devices)
    {
      if (other == device || !default_nccl_pool_supported(other))
      {
        continue;
      }

      auto& pool       = cudax::device_default_nccl_memory_pool(device);
      auto& other_pool = cudax::device_default_nccl_memory_pool(other);

      REQUIRE(&pool != &other_pool);
      REQUIRE(pool.get() != other_pool.get());
      REQUIRE(pool != other_pool);
    }
  }
}

C2H_TEST("device_default_nccl_memory_pool enables peer access", "[multi_gpu][nccl]")
{
  const auto device = require_supported_device();

  auto& pool = cudax::device_default_nccl_memory_pool(device);

  const auto peers = device.peers();

  if (peers.empty())
  {
    SKIP("Device has no peers");
  }

  for (auto peer : peers)
  {
    INFO("peer: " << peer.get());
    REQUIRE(pool.is_accessible_from(peer));
  }
}

C2H_TEST("device_default_nccl_memory_pool allocates", "[multi_gpu][nccl]")
{
  const auto device = require_supported_device();

  auto& pool  = cudax::device_default_nccl_memory_pool(device);
  auto stream = cudax::stream{device};

  SECTION("stream-ordered allocation round trip")
  {
    //! [device_default_nccl_memory_pool_allocate]
    constexpr std::size_t size = 1024;

    auto& nccl_pool = cudax::device_default_nccl_memory_pool(device);

    // Allocations are stream ordered against a stream on the pool's device.
    void* ptr = nccl_pool.allocate(stream, size);

    REQUIRE(ptr != nullptr);

    nccl_pool.deallocate(stream, ptr, size);
    stream.sync();
    //! [device_default_nccl_memory_pool_allocate]
  }

  SECTION("allocation is device accessible")
  {
    constexpr std::size_t size = sizeof(int);

    void* ptr = pool.allocate(stream, size);

    REQUIRE(ptr != nullptr);
    // A pointer drawn from a device pool must report as a device allocation.
    require_device_resident(ptr, device);

    pool.deallocate(stream, ptr, size);
    stream.sync();
  }
}

// Part of the contract of the default NCCL pool is that the pointers it hands out are usable
// wherever a pointer from `ncclMemAlloc()` would be, so that buffers can be registered with
// NCCL. The NCCL user-buffer-registration requirements are:
//
// - allocated through the VMM API with the shared handle type
//   `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`, and `CU_MEM_HANDLE_TYPE_FABRIC` on GPUs where
//   that is supported,
// - virtual head address aligned to at least the CUmem recommended granularity,
// - physical size a multiple of the CUmem recommended granularity.
//
// See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
C2H_TEST("device_default_nccl_memory_pool matches ncclMemAlloc properties", "[multi_gpu][nccl]")
{
  const auto device = require_supported_device();

  auto& pool = cudax::device_default_nccl_memory_pool(device);

  SECTION("pool is exportable through the handle types NCCL requires")
  {
    [[maybe_unused]] ::cuda::std::size_t exported{};

#if _CCCL_CTK_AT_LEAST(13, 1)
    exported = cuda::__driver::__mempoolGetAttribute(pool.get(), ::CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES);

    REQUIRE((exported & ::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0);

#endif // _CCCL_CTK_AT_LEAST(13, 1)

#if _CCCL_CTK_AT_LEAST(12, 4)
    const auto supported  = cuda::device_attributes::memory_pool_supported_handle_types(device);
    constexpr auto fabric = cuda::device_attributes::memory_pool_supported_handle_types_t::fabric;

    if ((supported & fabric) != 0)
    {
      REQUIRE((exported & ::CU_MEM_HANDLE_TYPE_FABRIC) != 0);
    }
#endif // _CCCL_CTK_AT_LEAST(12, 4)
  }

  SECTION("allocations are granularity aligned and granularity sized")
  {
    const std::size_t granularity = recommended_granularity(device);

    cudax::stream stream{device};

    // Deliberately request a size that is not a multiple of the granularity: the pool must
    // still hand back a suitably aligned head address backed by granularity-sized physical
    // memory.
    for (const std::size_t size : {std::size_t{1}, granularity / 2, granularity, granularity * 3 + 1})
    {
      INFO("requested size: " << size);

      void* ptr = pool.allocate(stream, size);

      REQUIRE(ptr != nullptr);
      REQUIRE(reinterpret_cast<std::uintptr_t>(ptr) % granularity == 0);

      // The allocation must lie inside a mapping that covers the whole request. `cuMemGetAddressRange`
      // reports the virtual range of the requested size rather than the granularity-rounded
      // physical backing, so it can only establish containment, not the physical size multiple.
      ::CUdeviceptr base = 0;
      std::size_t range  = 0;

      static auto* fn = driver_entry_point<decltype(::cuMemGetAddressRange)>("cuMemGetAddressRange");

      REQUIRE(fn(&base, &range, reinterpret_cast<::CUdeviceptr>(ptr)) == ::CUDA_SUCCESS);
      REQUIRE(base == reinterpret_cast<::CUdeviceptr>(ptr));
      REQUIRE(range >= size);

      pool.deallocate(stream, ptr, size);
      stream.sync();
    }
  }

  SECTION("allocations are VMM backed and readable/writable from the device")
  {
    const std::size_t granularity = recommended_granularity(device);

    cudax::stream stream{device};

    void* ptr = pool.allocate(stream, granularity);

    REQUIRE(ptr != nullptr);
    require_device_resident(ptr, device);
    // Memory must be writable and readable on the device it belongs to. Both wrappers throw on
    // failure, so reaching the comparison below is itself the success condition.
    constexpr unsigned char pattern = 0x2a;

    cuda::__driver::__memsetAsync(ptr, pattern, granularity, stream.get());

    std::vector<unsigned char> host(granularity, 0);

    cuda::__driver::__memcpyAsync(host.data(), ptr, granularity, stream.get());
    stream.sync();

    REQUIRE(std::all_of(host.begin(), host.end(), [](unsigned char value) {
      return value == static_cast<unsigned char>(pattern);
    }));

    pool.deallocate(stream, ptr, granularity);
    stream.sync();
  }
}
