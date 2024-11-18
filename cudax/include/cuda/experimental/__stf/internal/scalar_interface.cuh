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
struct owning_container_of;

template <typename T>
class scalar
{
public:
    T *addr;
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
  void stream_data_copy(const data_place& dst_memory_node, instance_id_t dst_instance_id, const data_place& src_memory_node, instance_id_t src_instance_id, cudaStream_t stream) override
  {
    assert(src_memory_node != dst_memory_node);

    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
    if (src_memory_node == data_place::host)
    {
      kind = cudaMemcpyHostToDevice;
    }

    if (dst_memory_node == data_place::host)
    {
      kind = cudaMemcpyDeviceToHost;
    }

    const scalar<T>& src_instance = this->instance(src_instance_id);
    const scalar<T>& dst_instance = this->instance(dst_instance_id);

    size_t sz = sizeof(T);

    cuda_safe_call(cudaMemcpyAsync((void*) dst_instance.addr, (void*) src_instance.addr, sz, kind, stream));
  }



  void stream_data_allocate(
    backend_ctx_untyped&, const data_place& memory_node, instance_id_t instance_id, ::std::ptrdiff_t& s, void**, cudaStream_t stream) override
  {
    scalar<T> &instance = this->instance(instance_id);
    T* base_ptr;

    if (memory_node == data_place::host)
    {
        // Fallback to a synchronous method as there is no asynchronous host allocation API
        cuda_safe_call(cudaStreamSynchronize(stream));
        cuda_safe_call(cudaHostAlloc(&base_ptr, sizeof(T), cudaHostAllocMapped));
    }
    else {
        cuda_safe_call(cudaMallocAsync(&base_ptr, sizeof(T), stream));
    }

    instance.addr = base_ptr;

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

template <typename T>
struct owning_container_of<scalar<T>>
{
  using type = T;
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
