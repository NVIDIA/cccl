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
 * @brief This implements a void data interface useful to implement STF
 * dependencies without actual data (e.g. to enforce task dependencies)
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
#include <cuda/experimental/__stf/stream/stream_data_interface.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::stf
{

template <typename T>
class shape_of;

template <typename T>
struct streamed_interface_of;

template <typename T>
struct graphed_interface_of;

class void_interface
{};

/**
 * @brief defines the shape of a void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends shape_of
 */
template <>
class shape_of<void_interface>
{
public:
  shape_of()                = default;
  shape_of(const shape_of&) = default;
  shape_of(const void_interface&)
      : shape_of<void_interface>()
  {}

  /// Mandatory method : defined the total number of elements in the shape
  size_t size() const
  {
    return 0;
  }
};

/**
 * @brief Data interface to manipulate the void interface in the CUDA stream backend
 */
class void_stream_interface : public stream_data_interface_simple<void_interface>
{
public:
  using base = stream_data_interface_simple<void_interface>;
  using base::shape_t;

  void_stream_interface(void_interface m)
      : base(::std::move(m))
  {}
  void_stream_interface(typename base::shape_t s)
      : base(s)
  {}

  /// Copy the content of an instance to another instance : this is a no-op
  void stream_data_copy(const data_place&, instance_id_t, const data_place&, instance_id_t, cudaStream_t) override {}

  /// Pretend we allocate an instance on a specific data place : we do not do any allocation here
  void stream_data_allocate(
    backend_ctx_untyped&, const data_place&, instance_id_t, ::std::ptrdiff_t& s, void**, cudaStream_t) override
  {
    // By filling a non negative number, we notify that the allocation was succesful
    s = 0;
  }

  /// Pretend we deallocate an instance (no-op)
  void stream_data_deallocate(backend_ctx_untyped&, const data_place&, instance_id_t, void*, cudaStream_t) override {}

  bool pin_host_memory(instance_id_t) override
  {
    // no-op
    return false;
  }

  void unpin_host_memory(instance_id_t) override {}
};

/**
 * @brief Define how the CUDA stream backend must manipulate this void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends streamed_interface_of
 */
template <>
struct streamed_interface_of<void_interface>
{
  using type = void_stream_interface;
};

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

/**
 * @brief A hash of the matrix
 */
template <>
struct hash<void_interface>
{
  ::std::size_t operator()(void_interface const&) const noexcept
  {
    return 42;
  }
};

} // end namespace cuda::experimental::stf
