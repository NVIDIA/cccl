//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Default memory pool resolved for the first time in the process while the calling thread is
// capturing: the pool attribute access must run in relaxed mode and leave the capture valid.
// Driver API only: the ccclrt fixture requires an empty driver context stack.

#include <cuda/__device/all_devices.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__memory_pool/memory_pool_base.h>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <testing.cuh>

#include "pool_availability.cuh"

namespace
{
struct capture_mode_case
{
  const char* name;
  ::CUstreamCaptureMode mode;
};

constexpr capture_mode_case capture_modes[] = {
  {"global", ::CU_STREAM_CAPTURE_MODE_GLOBAL},
  {"thread-local", ::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL},
};

template <class Fn>
Fn* driver_fn(const char* name)
{
  return reinterpret_cast<Fn*>(::cuda::__driver::__get_driver_entry_point(name));
}
::CUresult begin_capture(::CUstream stream, ::CUstreamCaptureMode mode)
{
  static auto fn = driver_fn<decltype(::cuStreamBeginCapture)>("cuStreamBeginCapture");
  return fn(stream, mode);
}
::CUresult end_capture(::CUstream stream, ::CUgraph* graph)
{
  static auto fn = driver_fn<decltype(::cuStreamEndCapture)>("cuStreamEndCapture");
  return fn(stream, graph);
}
size_t graph_node_count(::CUgraph graph)
{
  static auto fn = driver_fn<decltype(::cuGraphGetNodes)>("cuGraphGetNodes");
  size_t n       = 0;
  REQUIRE(fn(graph, nullptr, &n) == ::CUDA_SUCCESS);
  return n;
}
void destroy_graph(::CUgraph graph)
{
  static auto fn = driver_fn<decltype(::cuGraphDestroy)>("cuGraphDestroy");
  REQUIRE(fn(graph) == ::CUDA_SUCCESS);
}
// Refused under global/thread-local capture; accepted in relaxed mode.
::CUresult unsafe_pool_query(::CUmemoryPool pool)
{
  static auto fn       = driver_fn<decltype(::cuMemPoolGetAttribute)>("cuMemPoolGetAttribute");
  cuuint64_t threshold = 0;
  return fn(pool, ::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold);
}
} // namespace

C2H_CCCLRT_TEST("default memory pool resolved under stream capture", "[memory_resource][capture]")
{
  test::skip_if_unsupported_memory_pool<cuda::device_memory_pool_ref>();

  const cuda::device_ref dev = cuda::devices[0];
  ::CUmemLocation location{};
  location.type = ::CU_MEM_LOCATION_TYPE_DEVICE;
  location.id   = dev.get();

  for (const auto& c : capture_modes)
  {
    INFO("capture mode: " << c.name);
    const cuda::stream stream{dev};
    REQUIRE(begin_capture(stream.get(), c.mode) == ::CUDA_SUCCESS);

    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(dev);
    const ::cudaMemPool_t raw          = cuda::__get_default_memory_pool(location, ::CU_MEM_ALLOCATION_TYPE_PINNED);
    CCCLRT_REQUIRE(raw == pool.get());

    void* ptr = pool.allocate(stream, 1 << 20, ::cuda::mr::default_cuda_malloc_alignment);
    CCCLRT_REQUIRE(ptr != nullptr);
    pool.deallocate(stream, ptr, 1 << 20);

    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(stream.get(), &graph) == ::CUDA_SUCCESS);
    CCCLRT_CHECK(graph_node_count(graph) >= 1);
    destroy_graph(graph);

    CCCLRT_CHECK(cuda::memory_pool_attributes::release_threshold(pool.get()) != 0);
  }

  SECTION("the thread's capture mode is restored after the accessor returns")
  {
    const cuda::stream stream{dev};
    REQUIRE(begin_capture(stream.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    const ::cudaMemPool_t raw = cuda::__get_default_memory_pool(location, ::CU_MEM_ALLOCATION_TYPE_PINNED);

    // Thread mode was restored: the unsafe call is refused again.
    CCCLRT_CHECK(unsafe_pool_query(raw) == ::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);

    // The refused call invalidated the capture; end it.
    ::CUgraph graph = nullptr;
    CCCLRT_CHECK(end_capture(stream.get(), &graph) == ::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
  }
}
