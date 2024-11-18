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
 * @brief This implements a scalar interface to represent one value of a given type
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

template <typename T>
class scalar
{
public:
    // TODO operator () ...
    T val;
};

/**
 * @brief defines the shape of a scalar interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends shape_of
 */
template <typename T>
class shape_of<scalar<T>>
{
public:
  shape_of()                = default;
  shape_of(const shape_of&) = default;
  shape_of(const scalar<T>&)
      : shape_of<scalar<T>>()
  {}

  /// Mandatory method : defined the total number of elements in the shape
  size_t size() const
  {
    return sizeof(T);
  }
};

/**
 * @brief Data interface to manipulate the void interface in the CUDA stream backend
 */
template <typename T>
class scalar_stream_interface : public stream_data_interface_simple<scalar<T>>
{
public:
  using base = stream_data_interface_simple<scalar<T>>;
  using base::shape_t;

  scalar_stream_interface(scalar<T> val)
      : base(::std::move(val))
  {}
  scalar_stream_interface(typename base::shape_t s)
      : base(s)
  {}

  /// Copy the content of an instance to another instance : this is a no-op
  void stream_data_copy(const data_place&, instance_id_t, const data_place&, instance_id_t, cudaStream_t) override {}

  /// Pretend we allocate an instance on a specific data place : we do not do any allocation here
  void stream_data_allocate(
    backend_ctx_untyped&, const data_place&, instance_id_t, ::std::ptrdiff_t& s, void**, cudaStream_t) override
  {
    s = sizeof(T);
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
 * @brief Define how the CUDA stream backend must manipulate this scalar<T> interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends streamed_interface_of
 */
template <typename T>
struct streamed_interface_of<scalar<T>>
{
  using type = scalar_stream_interface<T>;
};

/**
 * @brief Data interface to manipulate the void interface in the CUDA graph backend
 */
template <typename T>
class scalar_graph_interface : public graph_data_interface<scalar<T>>
{
public:
  /// @brief Alias for the base class
  using base = graph_data_interface<scalar<T>>;
  /// @brief Alias for the shape type
  using base::shape_t;

  scalar_graph_interface(scalar<T> val)
      : base(mv(val))
  {}
  scalar_graph_interface(shape_of<scalar<T>> s)
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
    s = sizeof(T);
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
template <typename T>
struct graphed_interface_of<scalar<T>>
{
  using type = scalar_graph_interface<T>;
};

/**
 * @brief A hash of the content
 */
template <typename T>
struct hash<scalar<T>>
{
  ::std::size_t operator()(scalar<T> const&) const noexcept
  {
    return 16; // TODO !
  }
};

} // end namespace cuda::experimental::stf
