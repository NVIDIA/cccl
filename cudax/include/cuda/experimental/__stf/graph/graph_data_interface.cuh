//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of the graph_data_interface class which makes it
 *        possible to define data interfaces in the CUDA graph backend
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/graph/internal/event_types.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/internal/data_interface.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Data interface for the CUDA graph backend
 */
template <typename T>
class graph_data_interface : public data_impl_base<T>
{
public:
  using base    = data_impl_base<T>;
  using shape_t = typename base::shape_t;

  graph_data_interface(T p)
      : base(mv(p))
  {}

  graph_data_interface(shape_of<T> s)
      : base(mv(s))
  {}

  virtual cudaGraphNode_t graph_data_copy(
    cudaMemcpyKind kind,
    instance_id_t src_instance_id,
    instance_id_t dst_instance_id,
    cudaGraph_t graph,
    const cudaGraphNode_t* input_nodes,
    size_t input_cnt) = 0;

  // Returns prereq
  void data_copy(backend_ctx_untyped& ctx_,
                 const data_place& dst_memory_node,
                 instance_id_t dst_instance_id,
                 const data_place& src_memory_node,
                 instance_id_t src_instance_id,
                 event_list& prereqs) override
  {
    ::std::ignore = src_memory_node;
    ::std::ignore = dst_memory_node;
    assert(src_memory_node != dst_memory_node);

    cudaGraph_t graph  = ctx_.graph();
    size_t graph_stage = ctx_.stage();
    assert(graph && graph_stage != size_t(-1));

    const ::std::vector<cudaGraphNode_t> nodes = reserved::join_with_graph_nodes(ctx_, prereqs, graph_stage);

    // Let CUDA figure out from pointers
    cudaMemcpyKind kind = cudaMemcpyDefault;

    cudaGraphNode_t out = graph_data_copy(kind, src_instance_id, dst_instance_id, graph, nodes.data(), nodes.size());

    reserved::fork_from_graph_node(ctx_, out, graph, graph_stage, prereqs, "copy");
  }
};
} // end namespace cuda::experimental::stf
