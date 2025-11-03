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

/**
 * @brief View of an object of type `T`
 *
 * This is used to store a single value in a logical data.
 */
template <typename T>
struct scalar_view
{
  scalar_view() = default;
  scalar_view(T* _addr)
      : addr(_addr)
  {}

  T* addr;

  _CCCL_HOST_DEVICE T& operator*() const
  {
    return *addr;
  }
};

/**
 * @brief defines the shape of a scalar interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends shape_of
 */
template <typename T>
class shape_of<scalar_view<T>>
{
public:
  shape_of() = default;
  shape_of(const scalar_view<T>&)
      : shape_of<scalar_view<T>>()
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
class scalar_stream_interface : public stream_data_interface<scalar_view<T>>
{
public:
  using base = stream_data_interface<scalar_view<T>>;
  using typename base::shape_t;

  scalar_stream_interface(scalar_view<T> val)
      : base(::std::move(val))
  {}
  scalar_stream_interface(typename base::shape_t s)
      : base(s)
  {}

  /// Copy the content of an instance to another instance : this is a no-op
  void stream_data_copy(
    const data_place& dst_memory_node,
    instance_id_t dst_instance_id,
    const data_place& src_memory_node,
    instance_id_t src_instance_id,
    cudaStream_t stream) override
  {
    _CCCL_ASSERT(src_memory_node != dst_memory_node, "");

    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
    if (src_memory_node.is_host())
    {
      kind = cudaMemcpyHostToDevice;
    }

    if (dst_memory_node.is_host())
    {
      kind = cudaMemcpyDeviceToHost;
    }

    const scalar_view<T>& src_instance = this->instance(src_instance_id);
    const scalar_view<T>& dst_instance = this->instance(dst_instance_id);

    size_t sz = sizeof(T);

    cuda_safe_call(cudaMemcpyAsync((void*) dst_instance.addr, (void*) src_instance.addr, sz, kind, stream));
  }

  void data_allocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void**,
    event_list& prereqs) override
  {
    scalar_view<T>& instance = this->instance(instance_id);
    _CCCL_ASSERT(!memory_node.is_invalid(), "invalid memory node");

    s = sizeof(T);

    T* base_ptr   = static_cast<T*>(custom_allocator.allocate(bctx, memory_node, s, prereqs));
    instance.addr = base_ptr;
  }

  void data_deallocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    void*,
    event_list& prereqs) override
  {
    auto& local_desc = this->instance(instance_id);
    custom_allocator.deallocate(bctx, memory_node, prereqs, local_desc.addr, sizeof(T));
  }

  bool pin_host_memory(instance_id_t instance_id) override
  {
    auto& s = this->instance(instance_id);
    if (address_is_pinned(s.addr))
    {
      return false;
    }
    pin_memory(s.addr, 1);
    return true;
  }

  void unpin_host_memory(instance_id_t instance_id) override
  {
    auto& s = this->instance(instance_id);
    unpin_memory(s.addr);
  }
};

/**
 * @brief Define how the CUDA stream backend must manipulate this scalar_view<T> interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends streamed_interface_of
 */
template <typename T>
struct streamed_interface_of<scalar_view<T>>
{
  using type = scalar_stream_interface<T>;
};

/**
 * @brief Data interface to manipulate the void interface in the CUDA graph backend
 */
template <typename T>
class scalar_graph_interface : public graph_data_interface<scalar_view<T>>
{
public:
  /// @brief Alias for the base class
  using base = graph_data_interface<scalar_view<T>>;
  /// @brief Alias for the shape type
  using typename base::shape_t;

  scalar_graph_interface(scalar_view<T> val)
      : base(mv(val))
  {}
  scalar_graph_interface(shape_of<scalar_view<T>> s)
      : base(mv(s))
  {}

  void data_allocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& a,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void**,
    event_list& prereqs) override
  {
    s = sizeof(T);

    void* base_ptr = a.allocate(bctx, memory_node, s, prereqs);

    auto& local_desc = this->instance(instance_id);
    local_desc       = scalar_view<T>(static_cast<T*>(base_ptr));
  }

  void data_deallocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& a,
    const data_place& memory_node,
    instance_id_t instance_id,
    void*,
    event_list& prereqs) final
  {
    auto& local_desc = this->instance(instance_id);
    a.deallocate(bctx, memory_node, prereqs, local_desc.addr, sizeof(T));
  }

  cudaGraphNode_t graph_data_copy(
    cudaMemcpyKind kind,
    instance_id_t src_instance_id,
    instance_id_t dst_instance_id,
    cudaGraph_t graph,
    const cudaGraphNode_t* input_nodes,
    size_t input_cnt) override
  {
    const auto& src_instance = this->instance(src_instance_id);
    const auto& dst_instance = this->instance(dst_instance_id);

    T* src_ptr = src_instance.addr;
    T* dst_ptr = dst_instance.addr;

    cudaMemcpy3DParms cpy_params = {
      .srcArray = nullptr,
      .srcPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
      .srcPtr   = make_cudaPitchedPtr(src_ptr, sizeof(T), 1, 1),
      .dstArray = nullptr,
      .dstPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
      .dstPtr   = make_cudaPitchedPtr(dst_ptr, sizeof(T), 1, 1),
      .extent   = make_cudaExtent(sizeof(T), 1, 1),
      .kind     = kind};

    cudaGraphNode_t result;
    cuda_safe_call(cudaGraphAddMemcpyNode(&result, graph, input_nodes, input_cnt, &cpy_params));
    return result;
  }

  bool pin_host_memory(instance_id_t instance_id) override
  {
    auto s = this->instance(instance_id);
    if (address_is_pinned(s.addr))
    {
      return false;
    }
    pin_memory(s.addr, 1);
    return true;
  }

  void unpin_host_memory(instance_id_t) override
  {
    // no-op ... we unfortunately cannot unpin memory safely yet because
    // the graph may be executed a long time after this unpinning occurs.
  }
};

/**
 * @brief Define how the CUDA stream backend must manipulate this void interface
 *
 * Note that we specialize cuda::experimental::stf::shape_of to avoid ambiguous specialization
 *
 * @extends graphed_interface_of
 */
template <typename T>
struct graphed_interface_of<scalar_view<T>>
{
  using type = scalar_graph_interface<T>;
};

template <typename T>
struct owning_container_of<scalar_view<T>>
{
  using type = T;

  __host__ __device__ static void fill(scalar_view<T>& s, const T& val)
  {
    *s.addr = val;
  }

  __host__ __device__ static T get_value(const scalar_view<T>& s)
  {
    return *s.addr;
  }
};

/**
 * @brief A hash of the content
 */
template <typename T>
struct hash<scalar_view<T>>
{
  ::std::size_t operator()(scalar_view<T> const& s) const noexcept
  {
    return ::std::hash<T>{}(*s.addr);
  }
};
} // end namespace cuda::experimental::stf
