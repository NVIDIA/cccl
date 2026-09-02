//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DRIVER_DRIVER_API_CUH
#define _CUDAX__DRIVER_DRIVER_API_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/cstddef>

#  include <cuda.h>
#  include <cudaTypedefs.h>

#  include <cuda/std/__cccl/prologue.h>

// Get a driver function pointer, casting to the PFN typedef for type safety.
// Uses PFN_ typedefs from cudaTypedefs.h to avoid ABI mismatches caused by
// #define'd version aliases in cuda.h (e.g. #define cuFoo cuFoo_v2).
// The ## operator suppresses macro expansion of the function name, so this is
// safe even for names that are #define'd to versioned variants.
#  define _CUDAX_GET_DRIVER_FUNCTION(pfn_name, major, minor)    \
    reinterpret_cast<::PFN_##pfn_name##_v##major##0##minor##0>( \
      ::cuda::__driver::__get_driver_entry_point(#pfn_name, major, minor))

namespace cuda::experimental::__driver
{
// ── Graph: polymorphic add node ─────────────────────────────────────────────

#  if _CCCL_CTK_AT_LEAST(12, 2)

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddNode(
  ::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUgraphNodeParams* __params)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddNode, 12, 2);
  ::CUgraphNode __node{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a node to graph", &__node, __graph, __deps, __ndeps, __params);
  return __node;
}

#  endif // _CCCL_CTK_AT_LEAST(12, 2)

// ── Graph: memory allocation node ───────────────────────────────────────────

struct __graphAddMemAllocNodeResult
{
  ::CUgraphNode __node;
  ::CUdeviceptr __dptr;
};

[[nodiscard]] _CCCL_HOST_API inline __graphAddMemAllocNodeResult __graphAddMemAllocNode(
  ::CUgraph __graph,
  const ::CUgraphNode* __deps,
  ::cuda::std::size_t __ndeps,
  ::cuda::std::size_t __bytesize,
  int __device_id)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddMemAllocNode, 11, 4);
  ::CUgraphNode __node{};
  ::CUDA_MEM_ALLOC_NODE_PARAMS __params{};
  __params.poolProps.allocType   = ::CU_MEM_ALLOCATION_TYPE_PINNED;
  __params.poolProps.handleTypes = ::CU_MEM_HANDLE_TYPE_NONE;
  __params.poolProps.location    = {::CU_MEM_LOCATION_TYPE_DEVICE, __device_id};
  __params.bytesize              = __bytesize;

  ::CUmemAccessDesc __access_desc{};
  __access_desc.location = {::CU_MEM_LOCATION_TYPE_DEVICE, __device_id};
  __access_desc.flags    = ::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  __params.accessDescs     = &__access_desc;
  __params.accessDescCount = 1;

  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a memory allocation node to graph", &__node, __graph, __deps, __ndeps, &__params);
  return {__node, __params.dptr};
}

// ── Graph: memory free node ─────────────────────────────────────────────────

// ── Graph: memory free node (no-throw, for use in noexcept deallocate) ──────

struct __graphAddMemFreeNodeResult
{
  ::CUgraphNode __node;
  ::cudaError_t __status;
};

[[nodiscard]] _CCCL_HOST_API inline __graphAddMemFreeNodeResult __graphAddMemFreeNodeNoThrow(
  ::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUdeviceptr __dptr) noexcept
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddMemFreeNode, 11, 4);
  ::CUgraphNode __node{};
  auto __status = static_cast<::cudaError_t>(__driver_fn(&__node, __graph, __deps, __ndeps, __dptr));
  return {__node, __status};
}

// ── Graph: user object (ref-counted data lifetime tied to graph) ─────────────

_CCCL_HOST_API inline void __graphRetainUserObject(::CUgraph __graph, void* __ptr, ::CUhostFn __destroy)
{
  static auto __create_fn = _CUDAX_GET_DRIVER_FUNCTION(cuUserObjectCreate, 11, 3);
  static auto __retain_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphRetainUserObject, 11, 3);

  ::CUuserObject __obj{};
  ::cuda::__driver::__call_driver_fn(
    __create_fn, "Failed to create user object", &__obj, __ptr, __destroy, 1u, ::CU_USER_OBJECT_NO_DESTRUCTOR_SYNC);
  // CU_GRAPH_USER_OBJECT_MOVE transfers our reference to the graph without incrementing.
  // After this call, the graph owns the sole reference — do not release.
  ::cuda::__driver::__call_driver_fn(
    __retain_fn, "Failed to retain user object on graph", __graph, __obj, 1u, ::CU_GRAPH_USER_OBJECT_MOVE);
}

// ── Graph: conditional handle ───────────────────────────────────────────────

#  if _CCCL_CTK_AT_LEAST(12, 4)

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphConditionalHandle
__graphConditionalHandleCreate(::CUgraph __graph, ::CUcontext __ctx, unsigned int __default_val, unsigned int __flags)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphConditionalHandleCreate, 12, 3);
  ::CUgraphConditionalHandle __handle{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to create a conditional handle", &__handle, __graph, __ctx, __default_val, __flags);
  return __handle;
}

#  endif // _CCCL_CTK_AT_LEAST(12, 4)

// ── Graph: create ───────────────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraph __graphCreate()
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphCreate, 10, 0);
  ::CUgraph __graph{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to create graph", &__graph, 0u);
  return __graph;
}

// ── Graph: destroy (no-throw, for use in destructors) ───────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __graphDestroyNoThrow(::CUgraph __graph) noexcept
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphDestroy, 10, 0);
  return static_cast<::cudaError_t>(__driver_fn(__graph));
}

// ── Graph: clone ────────────────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraph __graphClone(::CUgraph __original)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphClone, 10, 0);
  ::CUgraph __clone{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to clone graph", &__clone, __original);
  return __clone;
}

// ── Graph: get node count ───────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t __graphGetNodeCount(::CUgraph __graph)
{
  static auto __driver_fn     = _CUDAX_GET_DRIVER_FUNCTION(cuGraphGetNodes, 10, 0);
  ::cuda::std::size_t __count = 0;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get graph node count", __graph, nullptr, &__count);
  return __count;
}

// ── Graph: instantiate ──────────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphExec __graphInstantiate(::CUgraph __graph, unsigned long long __flags = 0)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphInstantiateWithFlags, 11, 4);
  ::CUgraphExec __exec{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to instantiate graph", &__exec, __graph, __flags);
  return __exec;
}

// ── Graph: launch ───────────────────────────────────────────────────────────

_CCCL_HOST_API inline void __graphLaunch(::CUgraphExec __exec, ::CUstream __stream)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphLaunch, 10, 0);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to launch graph", __exec, __stream);
}

// ── Graph exec: destroy (no-throw, for use in destructors) ──────────────────

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __graphExecDestroyNoThrow(::CUgraphExec __exec) noexcept
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphExecDestroy, 10, 0);
  return static_cast<::cudaError_t>(__driver_fn(__exec));
}

// ── Graph: add empty node ───────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode
__graphAddEmptyNode(::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddEmptyNode, 10, 0);
  ::CUgraphNode __node{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add an empty node to graph", &__node, __graph, __deps, __ndeps);
  return __node;
}

// ── Graph: add dependencies ─────────────────────────────────────────────────

_CCCL_HOST_API inline void __graphAddDependencies(
  ::CUgraph __graph, const ::CUgraphNode* __from, const ::CUgraphNode* __to, ::cuda::std::size_t __ndeps)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddDependencies, 10, 0);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to add graph dependencies", __graph, __from, __to, __ndeps);
}

#  if _CCCL_CTK_AT_LEAST(12, 3)
_CCCL_HOST_API inline void __graphAddDependencies(
  ::CUgraph __graph,
  const ::CUgraphNode* __from,
  const ::CUgraphNode* __to,
  ::cuda::std::size_t __ndeps,
  const ::CUgraphEdgeData* __edge_data)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddDependencies, 12, 3);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add graph dependencies", __graph, __from, __to, __edge_data, __ndeps);
}
#  endif // _CCCL_CTK_AT_LEAST(12, 3)

// ── Graph node: get type ────────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNodeType __graphNodeGetType(::CUgraphNode __node)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphNodeGetType, 10, 0);
  ::CUgraphNodeType __type{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get graph node type", __node, &__type);
  return __type;
}

// ── Stream capture: begin capture to graph ──────────────────────────────────

#  if _CCCL_CTK_AT_LEAST(12, 3)

_CCCL_HOST_API inline void __streamBeginCaptureToGraph(
  ::CUstream __stream,
  ::CUgraph __graph,
  const ::CUgraphNode* __deps,
  ::cuda::std::size_t __ndeps,
  ::CUstreamCaptureMode __mode)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuStreamBeginCaptureToGraph, 12, 3);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to begin stream capture to graph", __stream, __graph, __deps, nullptr, __ndeps, __mode);
}

// ── Stream capture: get capture info ────────────────────────────────────────

struct __stream_capture_info
{
  ::CUstreamCaptureStatus __status;
  const ::CUgraphNode* __deps;
  const ::CUgraphEdgeData* __edge_data;
  ::cuda::std::size_t __ndeps;
};

[[nodiscard]] _CCCL_HOST_API inline __stream_capture_info
__streamGetCaptureInfo(::CUstream __stream, const ::CUgraphEdgeData** __edge_data_out = nullptr)
{
  __stream_capture_info __info{};
#    if _CCCL_CTK_AT_LEAST(12, 4)
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuStreamGetCaptureInfo, 12, 3);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to get stream capture info",
    __stream,
    &__info.__status,
    nullptr, // id_out
    nullptr, // graph_out
    &__info.__deps,
    &__info.__edge_data,
    &__info.__ndeps);
#    else
  _CCCL_ASSERT(__edge_data_out == nullptr, "Edge data requires CUDA Toolkit 12.4 or later");
  __info.__edge_data      = nullptr;
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuStreamGetCaptureInfo, 11, 3);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to get stream capture info",
    __stream,
    &__info.__status,
    nullptr, // id_out
    nullptr, // graph_out
    &__info.__deps,
    &__info.__ndeps);
#    endif
  return __info;
}

// ── Stream capture: end capture ─────────────────────────────────────────────

_CCCL_HOST_API inline void __streamEndCapture(::CUstream __stream, ::CUgraph* __graph_out)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuStreamEndCapture, 10, 0);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to end stream capture", __stream, __graph_out);
}

#  endif // _CCCL_CTK_AT_LEAST(12, 3)
} // namespace cuda::experimental::__driver

#  undef _CUDAX_GET_DRIVER_FUNCTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDAX__DRIVER_DRIVER_API_CUH
