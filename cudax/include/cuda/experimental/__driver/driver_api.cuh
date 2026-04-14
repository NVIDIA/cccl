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
#  define _CUDAX_GET_DRIVER_FUNCTION(pfn_name, major, minor)  \
    reinterpret_cast<PFN_##pfn_name##_v##major##0##minor##0>( \
      ::cuda::__driver::__get_driver_entry_point(#pfn_name, major, minor))

namespace cuda::experimental::__driver
{
// ── Graph: memset node ──────────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddMemsetNode(
  ::CUgraph __graph,
  const ::CUgraphNode* __deps,
  ::cuda::std::size_t __ndeps,
  ::CUdeviceptr __dst,
  ::cuda::std::size_t __pitch,
  unsigned int __value,
  unsigned int __element_size,
  ::cuda::std::size_t __width,
  ::cuda::std::size_t __height)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddMemsetNode, 10, 0);
  ::CUgraphNode __node{};
  ::CUDA_MEMSET_NODE_PARAMS __params{};
  __params.dst         = __dst;
  __params.pitch       = __pitch;
  __params.value       = __value;
  __params.elementSize = __element_size;
  __params.width       = __width;
  __params.height      = __height;
  ::CUcontext __ctx    = ::cuda::__driver::__ctxGetCurrent();
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a memset node to graph", &__node, __graph, __deps, __ndeps, &__params, __ctx);
  return __node;
}

// ── Graph: memcpy node (1-D) ────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddMemcpyNode1D(
  ::CUgraph __graph,
  const ::CUgraphNode* __deps,
  ::cuda::std::size_t __ndeps,
  ::CUdeviceptr __dst,
  ::CUdeviceptr __src,
  ::cuda::std::size_t __byte_count)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddMemcpyNode, 10, 0);
  ::CUgraphNode __node{};
  ::CUDA_MEMCPY3D __params{};
  __params.srcMemoryType = ::CU_MEMORYTYPE_UNIFIED;
  __params.srcDevice     = __src;
  __params.dstMemoryType = ::CU_MEMORYTYPE_UNIFIED;
  __params.dstDevice     = __dst;
  __params.WidthInBytes  = __byte_count;
  __params.Height        = 1;
  __params.Depth         = 1;
  ::CUcontext __ctx      = ::cuda::__driver::__ctxGetCurrent();
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a memcpy node to graph", &__node, __graph, __deps, __ndeps, &__params, __ctx);
  return __node;
}

// ── Graph: host node ────────────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddHostNode(
  ::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUhostFn __fn, void* __user_data)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddHostNode, 10, 0);
  ::CUgraphNode __node{};
  ::CUDA_HOST_NODE_PARAMS __params{};
  __params.fn       = __fn;
  __params.userData = __user_data;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a host node to graph", &__node, __graph, __deps, __ndeps, &__params);
  return __node;
}

// ── Graph: child graph node ─────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddChildGraphNode(
  ::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUgraph __child_graph)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddChildGraphNode, 10, 0);
  ::CUgraphNode __node{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a child graph node", &__node, __graph, __deps, __ndeps, __child_graph);
  return __node;
}

// ── Graph: event record node ────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode
__graphAddEventRecordNode(::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUevent __ev)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddEventRecordNode, 11, 1);
  ::CUgraphNode __node{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add an event record node to graph", &__node, __graph, __deps, __ndeps, __ev);
  return __node;
}

// ── Graph: event wait node ──────────────────────────────────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode
__graphAddEventWaitNode(::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUevent __ev)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddEventWaitNode, 11, 1);
  ::CUgraphNode __node{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add an event wait node to graph", &__node, __graph, __deps, __ndeps, __ev);
  return __node;
}

// ── Graph: conditional handle ───────────────────────────────────────────────

#  if _CCCL_CTK_AT_LEAST(12, 4) && _CCCL_CTK_BELOW(13, 0)

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphConditionalHandle
__graphConditionalHandleCreate(::CUgraph __graph, unsigned int __default_val, unsigned int __flags)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphConditionalHandleCreate, 12, 3);
  ::CUgraphConditionalHandle __handle{};
  ::CUcontext __ctx = ::cuda::__driver::__ctxGetCurrent();
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to create a conditional handle", &__handle, __graph, __ctx, __default_val, __flags);
  return __handle;
}

// ── Graph: generic add node (used for conditional nodes) ────────────────────

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddNode(
  ::CUgraph __graph, const ::CUgraphNode* __deps, ::cuda::std::size_t __ndeps, ::CUgraphNodeParams* __params)
{
  static auto __driver_fn = _CUDAX_GET_DRIVER_FUNCTION(cuGraphAddNode, 12, 2);
  ::CUgraphNode __node{};
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a node to graph", &__node, __graph, __deps, __ndeps, __params);
  return __node;
}

#  endif // _CCCL_CTK_AT_LEAST(12, 4) && _CCCL_CTK_BELOW(13, 0)

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
