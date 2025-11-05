//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_GRAPH_NODE_TYPE
#define __CUDAX_GRAPH_GRAPH_NODE_TYPE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! \enum graph_node_type
//!
//! \brief Represents the types of nodes that can exist in a CUDA graph.
//!
//! This enumeration defines various node types that can be used in CUDA graphs
//! to represent different operations or functionalities.
//!
//! \var graph_node_type::kernel
//! Represents a kernel execution node.
//!
//! \var graph_node_type::memcpy
//! Represents a memory copy operation node.
//!
//! \var graph_node_type::memset
//! Represents a memory set operation node.
//!
//! \var graph_node_type::host
//! Represents a host function execution node.
//!
//! \var graph_node_type::graph
//! Represents a nested graph node.
//!
//! \var graph_node_type::empty
//! Represents an empty node with no operation.
//!
//! \var graph_node_type::wait_event
//! Represents a node that waits for an event.
//!
//! \var graph_node_type::event_record
//! Represents a node that records an event.
//!
//! \var graph_node_type::semaphore_signal
//! Represents a node that signals an external semaphore.
//!
//! \var graph_node_type::semaphore_wait
//! Represents a node that waits on an external semaphore.
//!
//! \var graph_node_type::malloc
//! Represents a node that performs memory allocation.
//!
//! \var graph_node_type::free
//! Represents a node that performs memory deallocation.
//!
//! \var graph_node_type::batch_memop
//! Represents a node that performs a batch memory operation.
//!
//! \var graph_node_type::conditional
//! Represents a conditional execution node.
enum class graph_node_type : int
{
  kernel           = cudaGraphNodeTypeKernel,
  memcpy           = cudaGraphNodeTypeMemcpy,
  memset           = cudaGraphNodeTypeMemset,
  host             = cudaGraphNodeTypeHost,
  graph            = cudaGraphNodeTypeGraph,
  empty            = cudaGraphNodeTypeEmpty,
  wait_event       = cudaGraphNodeTypeWaitEvent,
  event_record     = cudaGraphNodeTypeEventRecord,
  semaphore_signal = cudaGraphNodeTypeExtSemaphoreSignal,
  semaphore_wait   = cudaGraphNodeTypeExtSemaphoreWait,
  malloc           = cudaGraphNodeTypeMemAlloc,
  free             = cudaGraphNodeTypeMemFree,
// batch_memop      = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP, // not exposed by the CUDA runtime

#if _CCCL_CTK_AT_LEAST(12, 8)
  conditional = cudaGraphNodeTypeConditional
#endif // _CCCL_CTK_AT_LEAST(12, 8)
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_GRAPH_NODE_TYPE
