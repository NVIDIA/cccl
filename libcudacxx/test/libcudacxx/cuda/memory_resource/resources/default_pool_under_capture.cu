//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Resolving a driver default memory pool lazily while the calling thread is inside a stream
// capture. `__get_default_memory_pool` applies the library's retention policy through a pool
// attribute read (and possibly write), which the driver refuses in global and thread-local
// capture modes; it must do so in a relaxed-capture window so the lookup succeeds and the capture
// stays valid. This test is deliberately the FIRST touch of the device default pool in its
// process, so the once-only path of `device_default_memory_pool` runs under capture.
//
// Driver entry points only (no cudart): the ccclrt fixture requires an empty driver context
// stack, which any runtime call would violate.

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

// Raw driver calls the library does not wrap, resolved through the same entry-point mechanism.
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
// A driver call that is unsafe under capture: accepted only in relaxed mode. Used to prove the
// thread's mode was restored after the accessor returned.
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
  const ::CUmemLocation location{::CU_MEM_LOCATION_TYPE_DEVICE, dev.get()};

  for (const auto& c : capture_modes)
  {
    INFO("capture mode: " << c.name);
    cuda::stream stream{dev};
    REQUIRE(begin_capture(stream.get(), c.mode) == ::CUDA_SUCCESS);

    // Lazy resolution under capture: the public once-only accessor (first touch in this
    // process on the first iteration) and the underlying helper (runs the attribute read on
    // every call). Both must succeed and leave the capture valid.
    cuda::device_memory_pool_ref& pool = cuda::device_default_memory_pool(dev);
    const ::cudaMemPool_t raw          = cuda::__get_default_memory_pool(location, ::CU_MEM_ALLOCATION_TYPE_PINNED);
    CCCLRT_REQUIRE(raw == pool.get());

    // Stream-ordered work after the resolution is still recorded into the graph.
    void* ptr = pool.allocate(stream, 1 << 20, ::cuda::mr::default_cuda_malloc_alignment);
    CCCLRT_REQUIRE(ptr != nullptr);
    pool.deallocate(stream, ptr, 1 << 20);

    ::CUgraph graph = nullptr;
    REQUIRE(end_capture(stream.get(), &graph) == ::CUDA_SUCCESS);
    CCCLRT_CHECK(graph_node_count(graph) >= 1);
    destroy_graph(graph);

    // The retention policy was applied for real (immediately, not recorded).
    CCCLRT_CHECK(cuda::memory_pool_attributes::release_threshold(pool.get()) != 0);
  }

  SECTION("the thread's capture mode is restored after the accessor returns")
  {
    cuda::stream stream{dev};
    REQUIRE(begin_capture(stream.get(), ::CU_STREAM_CAPTURE_MODE_GLOBAL) == ::CUDA_SUCCESS);

    const ::cudaMemPool_t raw = cuda::__get_default_memory_pool(location, ::CU_MEM_ALLOCATION_TYPE_PINNED);

    // Had the accessor left the thread in relaxed mode, this unsafe call would now be accepted.
    CCCLRT_CHECK(unsafe_pool_query(raw) == ::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);

    // That refused call invalidated the capture, as it should in global mode; end it.
    ::CUgraph graph = nullptr;
    CCCLRT_CHECK(end_capture(stream.get(), &graph) == ::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
  }
}
