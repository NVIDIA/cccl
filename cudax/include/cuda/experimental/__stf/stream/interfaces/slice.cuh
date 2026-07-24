//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/** @file
 *
 * @brief Defines a single- and multi-dimensional span/range/view/slice for layout purposes. To be replaced or
 * supplanted by `std::mdspan` in the future.
 *
 * Not much access functionality is defined or needed; only data layout is of importance.
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

#include <cuda/experimental/__stf/localization/composite_slice.cuh>
#include <cuda/experimental/__stf/stream/stream_data_interface.cuh>

namespace cuda::experimental::stf
{
/** @brief Contiguous memory interface. Supports multiple dimensions (compile-time chosen) and strides (run-time
 * chosen).
 */
template <typename T, typename... P>
class mdspan_stream_interface : public stream_data_interface<mdspan<T, P...>>
{
public:
  static constexpr size_t dimensions = mdspan<T, P...>::rank();
  using base                         = stream_data_interface<mdspan<T, P...>>;
  using typename base::element_type;
  using typename base::shape_t;

  using mutable_value_type = typename ::std::remove_const<T>::type;

  mdspan_stream_interface(T* p)
      : base(mdspan<T, P...>{p})
  {
    static_assert(dimensions == 0, "This constructor is reserved for 0-dimensional data.");
  }

  mdspan_stream_interface(mdspan<T, P...> m)
      : base(mv(m))
  {}

  mdspan_stream_interface(shape_t s)
      : base(mv(s))
  {}

  void stream_data_copy(const data_place&, instance_id_t, const data_place&, instance_id_t, cudaStream_t) override
  {
    // We are using data_copy instead of stream_data_copy in this interface
    assert(!"This should not be called directly.");
  }

  void data_allocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** extra_args,
    event_list& prereqs) override
  {
    //        fprintf(stderr, "SLICE data_allocate : data_place %ld - size<0>(b)T) = %ld * %d =
    //        %ld KB\n",
    //                int(memory_node), this->size<0>(common), sizeof(T),
    //                this->size<0>(common) * sizeof(T) / 1024);

    s                = this->shape.size() * sizeof(T);
    auto& local_desc = this->instance(instance_id);

    _CCCL_ASSERT(!memory_node.is_invalid(), "invalid data place");

    T* base_ptr;

    if (!memory_node.is_composite())
    {
      base_ptr   = static_cast<T*>(custom_allocator.allocate(bctx, memory_node, s, prereqs));
      local_desc = this->shape.create(base_ptr);
      return;
    }

    size_t total_size = this->shape.size();

    // Get the extents stored as a dim4
    const dim4 data_dims = this->shape.get_data_dims();

    // Slices are linearized with dimension 0 varying fastest; index_to_pos is
    // the inverse of dim4::get_index and is shared with the raw composite
    // allocation path so the two conventions cannot drift.
    static_assert(dimensions <= 4);
    auto delinearize = [data_dims](size_t ind) {
      // Rank-0 (scalar) slices have no coordinates (get_data_dims() reports
      // an extent of 0 there, so index_to_pos must not be used)
      if constexpr (dimensions == 0)
      {
        return pos4(0, 0, 0, 0);
      }
      else
      {
        return data_dims.index_to_pos(ind);
      }
    };

    auto [array,
          cached_prereqs] = bctx.get_composite_cache().get(memory_node, delinearize, total_size, sizeof(T), data_dims);
    prereqs.merge(mv(cached_prereqs));
    base_ptr = static_cast<T*>(array->get_base_ptr());

    // Store this localized array in the extra_args associated to the
    // data instance so that we can use it later in the deallocation
    // method.
    *extra_args = array.release();
    local_desc  = this->shape.create(base_ptr);
  }

  void data_deallocate(
    backend_ctx_untyped& bctx,
    block_allocator_untyped& custom_allocator,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* extra_args,
    event_list& prereqs) override
  {
    const size_t sz  = this->shape.size() * sizeof(T);
    auto& local_desc = this->instance(instance_id);
    // We can deallocate a copy of a logical data even if it was only accessible in read only mode
    auto ptr = const_cast<mutable_value_type*>(local_desc.data_handle());

    // TODO erase local_desc to avoid future reuse by mistake
    // local_desc = element_type();

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
    auto* array = static_cast<localized_array*>(extra_args);
    bctx.get_composite_cache().put(
      memory_node,
      ::std::unique_ptr<localized_array>(array),
      prereqs,
      this->shape.size(),
      sizeof(T),
      this->shape.get_data_dims());
  }

  void data_copy(backend_ctx_untyped& bctx,
                 const data_place& dst_memory_node,
                 instance_id_t dst_instance_id,
                 const data_place& src_memory_node,
                 instance_id_t src_instance_id,
                 event_list& prereqs) override
  {
    assert(src_memory_node != dst_memory_node);
    // We support dimensions up to 2, or higher if the slices are contiguous
    // static_assert(dimensions <= 2, "unsupported yet.");
    //_CCCL_ASSERT(dimensions <= 2, "unsupported yet.");

    // Host places borrow stream pools from the current device. For host-destination copies,
    // prefer the source place stream so stream/context affinity follows the data origin.
    const bool dst_is_host_like          = dst_memory_node.is_host() || dst_memory_node.is_managed();
    const bool src_is_host_like          = src_memory_node.is_host() || src_memory_node.is_managed();
    const data_place& stream_memory_node = (dst_is_host_like && !src_is_host_like) ? src_memory_node : dst_memory_node;
    const auto augmented_s = stream_memory_node.getDataStream(bctx.async_resources().get_place_resources());
    [[maybe_unused]] const auto active_stream_place = stream_memory_node.affine_exec_place().activate();
    auto op                                         = stream_async_op(bctx, augmented_s, prereqs);

    if (bctx.generate_event_symbols())
    {
      // TODO + d->get_symbol();
      op.set_symbol("slice copy " + src_memory_node.to_string() + "->" + dst_memory_node.to_string());
    }

    cudaStream_t s = augmented_s.stream;

    // Let CUDA figure out from pointers
    cudaMemcpyKind kind = cudaMemcpyDefault;

    /* Get size */
    auto& b                  = this->shape;
    const auto& src_instance = this->instance(src_instance_id);
    const auto& dst_instance = this->instance(dst_instance_id);

    /* We are copying so the destination will be changed, but from this
     * might be a constant variable (when using a read-only access). Having a T
     * type with a const qualifier is therefore possible, even if these API do not
     * want const pointers. */
    auto dst_ptr = const_cast<mutable_value_type*>(dst_instance.data_handle());
    auto src_ptr = src_instance.data_handle();
    assert(src_ptr);
    assert(dst_ptr);

    if constexpr (dimensions == 0)
    {
      // cudaMemcpyAsync is an overload set (cuda_runtime.h adds an alternate-spelling
      // wrapper), so it keeps the runtime-status cuda_try form.
      cuda_try(cudaMemcpyAsync(dst_ptr, src_ptr, sizeof(T), kind, s));
    }
    else if constexpr (dimensions == 1)
    {
      cuda_try(cudaMemcpyAsync(dst_ptr, src_ptr, b.extent(0) * sizeof(T), kind, s));
    }
    else if constexpr (dimensions == 2)
    {
      cuda_try<cudaMemcpy2DAsync>(
        dst_ptr,
        dst_instance.stride(1) * sizeof(T),
        src_ptr,
        src_instance.stride(1) * sizeof(T),
        b.extent(0) * sizeof(T),
        b.extent(1),
        kind,
        s);
    }
    else
    {
      // We only support higher dimensions if they are contiguous !
      if ((contiguous_dims(src_instance) == dimensions) && (contiguous_dims(dst_instance) == dimensions))
      {
        cuda_try(cudaMemcpyAsync(dst_ptr, src_ptr, b.size() * sizeof(T), kind, s));
      }
      else
      {
        _CCCL_ASSERT(dimensions == 2, "Higher dimensions not supported.");
      }
    }

    prereqs = op.end(bctx);
  }

  bool pin_host_memory(instance_id_t instance_id) override
  {
    auto s = this->instance(instance_id);
    return s.data_handle() && pin(s);
  }

  void unpin_host_memory(instance_id_t instance_id) override
  {
    unpin(this->instance(instance_id));
  }

  ::std::optional<cudaMemoryType> get_memory_type(instance_id_t instance_id) override
  {
    const auto attributes = cuda_try<cudaPointerGetAttributes>(this->instance(instance_id).data_handle());
    // Implicitly converted to an optional
    return attributes.type;
  }
};

template <typename T>
struct streamed_interface_of;

template <typename T, typename... P>
struct streamed_interface_of<mdspan<T, P...>>
{
  using type = mdspan_stream_interface<T, P...>;
};
} // namespace cuda::experimental::stf
