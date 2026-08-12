// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/detail/device_memory_resource.cuh>

#include <cuda/__memory_resource/memory_resource_base.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__stream/stream_ref.h>

#include <c2h/catch2_test_helper.h>

struct device_memory_resource
    : cub::detail::device_memory_resource
    , ::cuda::mr::memory_resource_base<device_memory_resource>
{
  cudaStream_t target_stream = nullptr;
  size_t* bytes_allocated    = nullptr;
  size_t* bytes_deallocated  = nullptr;

  device_memory_resource() = default;

  device_memory_resource(cudaStream_t stream, size_t* alloc, size_t* dealloc)
      : target_stream(stream)
      , bytes_allocated(alloc)
      , bytes_deallocated(dealloc)
  {}

  void* allocate_sync(size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous allocation");
    return nullptr;
  }

  void deallocate_sync(void* /* ptr */, size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous deallocation");
  }

  void* allocate(cuda::stream_ref stream, size_t bytes, size_t /* alignment */)
  {
    return allocate(stream, bytes);
  }

  void* allocate(cuda::stream_ref stream, size_t bytes)
  {
    REQUIRE(target_stream == stream.get());

    if (bytes_allocated)
    {
      *bytes_allocated += bytes;
    }
    return cub::detail::device_memory_resource::allocate(stream, bytes);
  }

  void deallocate(const cuda::stream_ref stream, void* ptr, size_t bytes, size_t /* alignment */)
  {
    deallocate(stream, ptr, bytes);
  }

  void deallocate(const cuda::stream_ref stream, void* ptr, size_t bytes)
  {
    REQUIRE(target_stream == stream.get());

    if (bytes_deallocated)
    {
      *bytes_deallocated += bytes;
    }
    cub::detail::device_memory_resource::deallocate(stream, ptr, bytes);
  }

  bool operator==(const device_memory_resource& rhs) const
  {
    return target_stream == rhs.target_stream && bytes_allocated == rhs.bytes_allocated
        && bytes_deallocated == rhs.bytes_deallocated;
  }
  bool operator!=(const device_memory_resource& rhs) const
  {
    return !(*this == rhs);
  }
};
static_assert(::cuda::mr::resource<device_memory_resource>);

struct throwing_memory_resource
{
  void* allocate_sync(size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous allocation");
    return nullptr;
  }

  void deallocate_sync(void* /* ptr */, size_t /* bytes */, size_t /* alignment */)
  {
    FAIL("CUB shouldn't use synchronous deallocation");
  }

  void* allocate(cuda::stream_ref /* stream */, size_t /* bytes */, size_t /* alignment */)
  {
    throw "test";
  }

  void* allocate(cuda::stream_ref /* stream */, size_t /* bytes */)
  {
    throw "test";
  }

  void deallocate(cuda::stream_ref /* stream */, void* /* ptr */, size_t /* bytes */, size_t /* alignment*/)
  {
    throw "test";
  }

  void deallocate(cuda::stream_ref /* stream */, void* /* ptr */, size_t /* bytes */)
  {
    throw "test";
  }

  bool operator==(const throwing_memory_resource&) const
  {
    return true;
  }
  bool operator!=(const throwing_memory_resource&) const
  {
    return false;
  }
};
static_assert(::cuda::mr::resource<throwing_memory_resource>);

struct device_side_memory_resource
{
  void* ptr{};
  size_t* bytes_allocated   = nullptr;
  size_t* bytes_deallocated = nullptr;

  __host__ __device__ void* allocate_sync(size_t /* bytes */, size_t /* alignment */)
  {
    cuda::std::terminate();
  }

  __host__ __device__ void deallocate_sync(void* /* ptr */, size_t /* bytes */, size_t /* alignment */)
  {
    cuda::std::terminate();
  }

  __host__ __device__ void* allocate(cuda::stream_ref stream, size_t bytes, size_t /* alignment */)
  {
    return allocate(stream, bytes);
  }

  __host__ __device__ void* allocate(cuda::stream_ref /* stream */, size_t bytes)
  {
    if (bytes_allocated)
    {
      *bytes_allocated += bytes;
    }
    return static_cast<void*>(static_cast<char*>(ptr) + *bytes_allocated);
  }

  __host__ __device__ void deallocate(const cuda::stream_ref /* stream */, void* /* ptr */, size_t bytes)
  {
    if (bytes_deallocated)
    {
      *bytes_deallocated += bytes;
    }
  }

  __host__ __device__ void
  deallocate(const cuda::stream_ref /* stream */, void* /* ptr */, size_t bytes, size_t /* alignment */)
  {
    if (bytes_deallocated)
    {
      *bytes_deallocated += bytes;
    }
  }

  bool operator==(const device_side_memory_resource& rhs) const
  {
    return ptr == rhs.ptr && bytes_allocated == rhs.bytes_allocated && bytes_deallocated == rhs.bytes_deallocated;
  }
  bool operator!=(const device_side_memory_resource& rhs) const
  {
    return !(*this == rhs);
  }
};
static_assert(::cuda::mr::resource<device_side_memory_resource>);
