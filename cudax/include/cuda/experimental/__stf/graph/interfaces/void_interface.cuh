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
 *
 * @brief This implements a void data interface over the graph_ctx backend
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

#include <cuda/experimental/__stf/graph/graph_data_interface.cuh>
#include <cuda/experimental/__stf/internal/void_interface.cuh>

namespace cuda::experimental::stf
{
template <typename T>
struct graphed_interface_of;

/**
 * @brief Data interface to manipulate the void interface in the CUDA graph backend
 */
class void_graph_interface : public graph_data_interface<void_interface>
{
public:
  /// @brief Alias for the base class
  using base = graph_data_interface<void_interface>;
  /// @brief Alias for the shape type
  using base::shape_t;

  void_graph_interface(void_interface s)
      : base(mv(s))
  {}
  void_graph_interface(shape_of<void_interface> s)
      : base(mv(s))
  {}

  void data_allocate(
    backend_ctx_untyped&,
    block_allocator_untyped&,
    const data_place&,
    instance_id_t,
    ::std::ptrdiff_t& s,
    void**,
    event_list&) override
  {
    s = 0;
  }

  void data_deallocate(
    backend_ctx_untyped&, block_allocator_untyped&, const data_place&, instance_id_t, void*, event_list&) final
  {}

  cudaGraphNode_t graph_data_copy(
    cudaMemcpyKind,
    instance_id_t,
    instance_id_t,
    cudaGraph_t graph,
    const cudaGraphNode_t* input_nodes,
    size_t input_cnt) override
  {
    cudaGraphNode_t dummy;
    cuda_safe_call(cudaGraphAddEmptyNode(&dummy, graph, input_nodes, input_cnt));
    return dummy;
  }

  bool pin_host_memory(instance_id_t) override
  {
    // no-op
    return false;
  }

  void unpin_host_memory(instance_id_t) override {}

  /* This helps detecting when we are manipulating a void data interface, so
   * that we can optimize useless stages such as allocations or copies */
  bool is_void_interface() const override final
  {
    return true;
  }
};

/**
 * @brief Define how the CUDA stream backend must manipulate this void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends graphed_interface_of
 */
template <>
struct graphed_interface_of<void_interface>
{
  using type = void_graph_interface;
};
} // end namespace cuda::experimental::stf
