//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__GRAPH_CHILD_GRAPH_CUH
#define _CUDAX__GRAPH_CHILD_GRAPH_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/experimental/__driver/driver_api.cuh>
#  include <cuda/experimental/__graph/graph_builder.cuh>
#  include <cuda/experimental/__graph/graph_builder_ref.cuh>
#  include <cuda/experimental/__graph/graph_node_ref.cuh>
#  include <cuda/experimental/__graph/path_builder.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! \brief Adds a child graph node to a CUDA graph path.
//!
//! The entire subgraph described by \p __child is embedded as a single node in the parent
//! graph.  All nodes in the child graph execute before any successor of the new child-graph
//! node.
//!
//! \param __pb    Path builder to insert the node into.
//! \param __child A `graph_builder_ref` whose underlying graph will become the child.
//! \return A `graph_node_ref` for the newly added child-graph node.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_HOST_API inline graph_node_ref insert_child_graph(path_builder& __pb, graph_builder_ref __child)
{
  auto __deps = __pb.get_dependencies();
  ::CUgraphNodeParams __params{};
  __params.type        = ::CU_GRAPH_NODE_TYPE_GRAPH;
  __params.graph.graph = __child.get();
  auto __node          = ::cuda::experimental::__driver::__graphAddNode(
    __pb.get_native_graph_handle(), __deps.data(), __deps.size(), &__params);

  __pb.__clear_and_set_dependency_node(__node);
  return graph_node_ref{__node, __pb.get_native_graph_handle()};
}

#  if _CCCL_CTK_AT_LEAST(12, 9)
//! \brief Adds a child graph node to a CUDA graph path, transferring ownership.
//!
//! The child graph is moved into the parent graph node. After this call, the
//! `graph_builder` is left in a null state and the parent graph owns the child's
//! lifetime.
//!
//! \param __pb    Path builder to insert the node into.
//! \param __child An rvalue `graph_builder` whose graph will be moved into the parent.
//! \return A `graph_node_ref` for the newly added child-graph node.
//! \throws cuda::std::cuda_error if node creation fails.
_CCCL_HOST_API inline graph_node_ref insert_child_graph(path_builder& __pb, graph_builder&& __child)
{
  auto __deps = __pb.get_dependencies();
  ::CUgraphNodeParams __params{};
  __params.type            = ::CU_GRAPH_NODE_TYPE_GRAPH;
  __params.graph.graph     = __child.get();
  __params.graph.ownership = ::CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE;
  auto __node              = ::cuda::experimental::__driver::__graphAddNode(
    __pb.get_native_graph_handle(), __deps.data(), __deps.size(), &__params);

  (void) __child.release();

  __pb.__clear_and_set_dependency_node(__node);
  return graph_node_ref{__node, __pb.get_native_graph_handle()};
}
#  endif // _CCCL_CTK_AT_LEAST(12, 9)
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CTK_AT_LEAST(12, 2)

#endif // _CUDAX__GRAPH_CHILD_GRAPH_CUH
