//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_GRAPH_MEMORY_RESOURCE_CUH
#define _CUDAX__GRAPH_GRAPH_MEMORY_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/cstddef>

#  include <cuda/experimental/__driver/driver_api.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief A memory resource that allocates and frees device memory as CUDA graph nodes.
//!
//! Constructed with a device_ref that determines where memory is allocated.
//! Inserts `cuGraphAddMemAllocNode` and `cuGraphAddMemFreeNode` nodes into a graph
//! via a path_builder. Also supports stream-based deallocation for memory that outlives
//! the graph execution.
//!
struct graph_memory_resource
{
  //! @brief Construct a graph memory resource for the specified device.
  //! @param __dev The device on which memory will be allocated.
  _CCCL_HOST_API explicit graph_memory_resource(device_ref __dev) noexcept
      : __dev_(__dev)
  {}

  //! @brief Insert a memory allocation node into the graph.
  //! @param __pb The path builder to insert the alloc node into.
  //! @param __size Number of bytes to allocate.
  //! @param __alignment Alignment requirement (unused by the CUDA graph alloc API, reserved).
  //! @return Device pointer to the allocated memory.
  _CCCL_HOST_API void* allocate(path_builder& __pb,
                                ::cuda::std::size_t __size,
                                ::cuda::std::size_t __alignment = cuda::mr::default_cuda_malloc_alignment)
  {
    (void) __alignment;

    if (__size == 0)
    {
      return nullptr;
    }

    auto __deps           = __pb.get_dependencies();
    auto [__node, __dptr] = ::cuda::experimental::__driver::__graphAddMemAllocNode(
      __pb.get_native_graph_handle(), __deps.data(), __deps.size(), __size, __dev_.get());

    __pb.__clear_and_set_dependency_node(__node);
    return reinterpret_cast<void*>(__dptr);
  }

  //! @brief Insert a memory free node into the graph.
  //! @param __pb The path builder to insert the free node into.
  //! @param __ptr Device pointer previously returned by allocate().
  //! @param __size Number of bytes (unused, kept for interface symmetry).
  //! @param __alignment Alignment (unused, kept for interface symmetry).
  _CCCL_HOST_API void deallocate(
    path_builder& __pb,
    void* __ptr,
    ::cuda::std::size_t __size      = 0,
    ::cuda::std::size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment) noexcept
  {
    (void) __size;
    (void) __alignment;

    if (__ptr == nullptr)
    {
      return;
    }

    auto __deps          = __pb.get_dependencies();
    auto [__node, __err] = ::cuda::experimental::__driver::__graphAddMemFreeNodeNoThrow(
      __pb.get_native_graph_handle(), __deps.data(), __deps.size(), reinterpret_cast<::CUdeviceptr>(__ptr));
    _CCCL_ASSERT(__err == ::cudaSuccess, "Failed to add a memory free node to graph");

    __pb.__clear_and_set_dependency_node(__node);
  }

  //! @brief Free device memory asynchronously on a stream.
  //! @param __stream The stream on which to free the memory.
  //! @param __ptr Device pointer previously returned by allocate().
  //! @param __size Number of bytes (unused, kept for interface symmetry).
  //! @param __alignment Alignment (unused, kept for interface symmetry).
  _CCCL_HOST_API void deallocate(
    ::cuda::stream_ref __stream,
    void* __ptr,
    ::cuda::std::size_t __size      = 0,
    ::cuda::std::size_t __alignment = ::cuda::mr::default_cuda_malloc_alignment)
  {
    (void) __size;
    (void) __alignment;

    if (__ptr == nullptr)
    {
      return;
    }

    _CCCL_ASSERT_CUDA_API(
      ::cuda::__driver::__freeAsyncNoThrow,
      "graph_memory_resource::deallocate failed",
      reinterpret_cast<::CUdeviceptr>(__ptr),
      __stream.get());
  }

  //! @brief Returns the device this resource allocates on.
  [[nodiscard]] _CCCL_HOST_API device_ref device() const noexcept
  {
    return __dev_;
  }

  //! @brief Enables the \c device_accessible property.
  _CCCL_HOST_API friend constexpr void get_property(graph_memory_resource const&, ::cuda::mr::device_accessible) noexcept
  {}

  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

  _CCCL_HOST_API friend bool operator==(const graph_memory_resource& __lhs, const graph_memory_resource& __rhs) noexcept
  {
    return __lhs.__dev_ == __rhs.__dev_;
  }

  _CCCL_HOST_API friend bool operator!=(const graph_memory_resource& __lhs, const graph_memory_resource& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

private:
  device_ref __dev_;
};
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 2)

#endif // _CUDAX__GRAPH_GRAPH_MEMORY_RESOURCE_CUH
