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
 * @brief Implementation of the slice data interface in the `graph_ctx` backend
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
#include <cuda/experimental/__stf/localization/composite_slice.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Slice data interface implementation over CUDA graphs.
 *
 * @tparam T Element type for the slice
 * @tparam dimensions rank of the slice
 *
 * The type manipulated will be `slice<T, dimensions>`.
 */
template <typename T, size_t dimensions = 1>
class slice_graph_interface : public graph_data_interface<slice<T, dimensions>>
{
public:
  /// @brief Alias for the base class
  using base = graph_data_interface<slice<T, dimensions>>;
  /// @brief Alias for the shape type
  using typename base::shape_t;

  using mutable_value_type = typename ::std::remove_const<T>::type;

  /**
   * @brief Constructor from slice
   *
   * @param s either a slice or the shape of a slice
   */
  slice_graph_interface(slice<T, dimensions> s)
      : base(mv(s))
  {}

  /**
   * @brief Constructor from shape
   *
   * @param s either a slice or the shape of a slice
   */
  slice_graph_interface(shape_of<slice<T, dimensions>> s)
      : base(mv(s))
  {}

  /// @brief Implementation of interface primitive
  void data_allocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** extra_args,
    event_list& prereqs) override
  {
    s                = this->shape.size() * sizeof(T);
    auto& local_desc = this->instance(instance_id);

    if (!memory_node.is_composite())
    {
      void* base_ptr = custom_allocator.allocate(bctx, memory_node, s, prereqs);
      local_desc     = this->shape.create(static_cast<T*>(base_ptr));
      return;
    }

    exec_place_grid grid = memory_node.get_grid();
    size_t total_size    = this->shape.size();

    // position (x,y,z,t) on (nx,ny,nz,nt)
    // * index = x + nx*y + nx*ny*z + nx*ny*nz*t
    // * index = x + nx(y + ny(z + nz*t))
    // So we can compute x, y, z, t from the index
    // * x := index % nx;
    // * index - x = nx(y + ny(z + nz*t))
    // * (index - x)/nx = y +  ny(z + nz*t))
    // So, y := ( (index - x)/nx ) %ny
    // index = x + nx(y + ny(z + nz*t))
    // (index -  x)/nx = y + ny(z+nz*t)
    // (index -  x)/nx - y = ny(z+nz*t)
    // ( (index -  x)/nx - y)/ny  = z+nz*t
    // So, z := (( (index -  x)/nx - y)/ny) % nz
    // ( (index -  x)/nx - y)/ny - z  = nz*t
    // ( ( (index -  x)/nx - y)/ny - z ) / nz = t
    auto delinearize = [&](size_t ind) {
      static_assert(dimensions <= 4);

      size_t nx, ny, nz;
      size_t x = 0, y = 0, z = 0, t = 0;
      if constexpr (dimensions >= 1)
      {
        nx = this->shape.extent(0);
        x  = ind % nx;
      }
      if constexpr (dimensions >= 2)
      {
        ny = this->shape.extent(1);
        y  = ((ind - x) / nx) % ny;
      }
      if constexpr (dimensions >= 3)
      {
        nz = this->shape.extent(2);
        z  = (((ind - x) / nx - y) / ny) % nz;
      }

      if constexpr (dimensions >= 4)
      {
        t = (((ind - x) / nx - y) / ny - z) / nz;
      }

      return pos4(x, y, z, t);
    };

    // Get the extents stored as a dim4
    const dim4 data_dims = this->shape.get_data_dims();

    auto array = bctx.get_composite_cache().get(
      memory_node, memory_node.get_partitioner(), delinearize, total_size, sizeof(T), data_dims);
    assert(array);
    array->merge_into(prereqs);
    T* base_ptr = static_cast<T*>(array->get_base_ptr());

    // Store this localized array in the extra_args associated to the
    // data instance so that we can use it later in the deallocation
    // method.
    *extra_args = array.release();
    local_desc  = this->shape.create(base_ptr);
  }

  /// @brief Implementation of interface primitive
  void data_deallocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* extra_args,
    event_list& prereqs) final
  {
    const size_t sz  = this->shape.size() * sizeof(T);
    auto& local_desc = this->instance(instance_id);
    // We can deallocate a copy of a logical data even if it was only accessible in read only mode
    auto ptr = const_cast<mutable_value_type*>(local_desc.data_handle());

    // TODO find a way to erase this variable to facilitate debugging
    // local_desc = slice<T, dimensions>(); // optional, helps with debugging

    if (!memory_node.is_composite())
    {
      custom_allocator.deallocate(bctx, memory_node, prereqs, ptr, sz);
      return;
    }

    assert(extra_args);

    // To properly erase a composite data, we would need to synchronize the
    // calling thread with all events because the current CUDA VMM facility
    // does not have an asynchronous counterpart. Instead, we put the
    // localized_array object into a cache, which will speedup the
    // allocation of identical arrays, if any.
    // This cached array is only usable once the prereqs of this deallocation are fulfilled.
    auto* array = static_cast<reserved::localized_array*>(extra_args);
    bctx.get_composite_cache().put(::std::unique_ptr<reserved::localized_array>(array), prereqs);
  }

  /// @brief Implementation of interface primitive
  cudaGraphNode_t graph_data_copy(
    cudaMemcpyKind kind,
    instance_id_t src_instance_id,
    instance_id_t dst_instance_id,
    cudaGraph_t graph,
    const cudaGraphNode_t* input_nodes,
    size_t input_cnt) override
  {
    // static_assert(dimensions <= 2, "Unsupported yet.");
    const auto& b            = this->shape;
    const auto& src_instance = this->instance(src_instance_id);
    const auto& dst_instance = this->instance(dst_instance_id);

    cudaMemcpy3DParms cpy_params;

    /* We are copying so the destination will be changed, but from this
     * might be a constant variable (when using a read-only access). Having a T
     * type with a const qualifier is therefore possible, even if these API do not
     * want const pointers. */
    auto dst_ptr = const_cast<mutable_value_type*>(dst_instance.data_handle());

    // make_cudaPitchedPtr wants non const pointer
    auto src_ptr = const_cast<mutable_value_type*>(src_instance.data_handle());

    if constexpr (dimensions == 0)
    {
      cpy_params = {
        .srcArray = nullptr,
        .srcPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .srcPtr   = make_cudaPitchedPtr(src_ptr, sizeof(T), 1, 1),
        .dstArray = nullptr,
        .dstPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .dstPtr   = make_cudaPitchedPtr(dst_ptr, sizeof(T), 1, 1),
        .extent   = make_cudaExtent(sizeof(T), 1, 1),
        .kind     = kind};
    }
    else if constexpr (dimensions == 1)
    {
      cpy_params = {
        .srcArray = nullptr,
        .srcPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .srcPtr   = make_cudaPitchedPtr(src_ptr, b.extent(0) * sizeof(T), b.extent(0), 1),
        .dstArray = nullptr,
        .dstPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .dstPtr   = make_cudaPitchedPtr(dst_ptr, b.extent(0) * sizeof(T), b.extent(0), 1),
        .extent   = make_cudaExtent(b.extent(0) * sizeof(T), 1, 1),
        .kind     = kind};
    }
    else if constexpr (dimensions == 2)
    {
      cpy_params = {
        .srcArray = nullptr,
        .srcPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .srcPtr   = make_cudaPitchedPtr(src_ptr, src_instance.stride(1) * sizeof(T), b.extent(0), b.extent(1)),
        .dstArray = nullptr,
        .dstPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .dstPtr   = make_cudaPitchedPtr(dst_ptr, dst_instance.stride(1) * sizeof(T), b.extent(0), b.extent(1)),
        .extent   = make_cudaExtent(b.extent(0) * sizeof(T), b.extent(1), 1),
        .kind     = kind};
    }
    else
    {
      // For more than 2D
      EXPECT(contiguous_dims(b) == dimensions, "Cannot handle non-contiguous multidimensional array above 2D.");

      cpy_params = {
        .srcArray = nullptr,
        .srcPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .srcPtr   = make_cudaPitchedPtr(src_ptr, b.size() * sizeof(T), b.size(), 1),
        .dstArray = nullptr,
        .dstPos   = make_cudaPos(size_t(0), size_t(0), size_t(0)),
        .dstPtr   = make_cudaPitchedPtr(dst_ptr, b.size() * sizeof(T), b.size(), 1),
        .extent   = make_cudaExtent(b.size() * sizeof(T), 1, 1),
        .kind     = kind};
    }

    cudaGraphNode_t result;
    cuda_safe_call(cudaGraphAddMemcpyNode(&result, graph, input_nodes, input_cnt, &cpy_params));

    return result;
  }

  /// @brief Implementation of interface primitive
  bool pin_host_memory(instance_id_t instance_id) override
  {
    auto s = this->instance(instance_id);
    return s.data_handle() != nullptr && pin(s);
  }

  /// @brief Implementation of interface primitive
  void unpin_host_memory(instance_id_t /*instance_id*/) override
  {
    // no-op ... we unfortunately cannot unpin memory safely yet because
    // the graph may be executed a long time after this unpinning occurs.
    /// assert(memory_node.is_host());

    /// const auto& common = this->common;
    /// const auto& per_inst = this->instance(instance_id);

    /// slice<T, dimensions> s = slice<T, dimensions>(common, per_inst);

    /// unpin(s);
  }
};

/**
 * @brief Given a type, defines `type` as the corresponding data interface over graphs.
 *
 * @tparam T
 */
template <typename T>
struct graphed_interface_of;

template <typename T, typename... P>
struct graphed_interface_of<mdspan<T, P...>>
{
  using type = slice_graph_interface<T, mdspan<T, P...>::rank()>;
};
} // namespace cuda::experimental::stf
