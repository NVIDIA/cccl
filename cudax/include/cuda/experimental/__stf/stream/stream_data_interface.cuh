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
 * @brief Implementation of the stream_data_interface class which makes it
 *        possible to define data interfaces in the CUDA stream backend
 *
 * The stream_data_interface_simple provides a simpler interface compared to stream_data_interface
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

#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/internal/data_interface.cuh>
#include <cuda/experimental/__stf/stream/internal/event_types.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Data interface for the CUDA streams backend.
 *
 * This is similar to stream_data_interface_simple but relies on a more
 * sophisticated allocator scheme that caches memory allocations.
 */
template <typename T>
class stream_data_interface : public data_impl_base<T>
{
public:
  using base = data_impl_base<T>;
  using typename base::shape_t;

  stream_data_interface(T object)
      : base(mv(object))
  {}

  stream_data_interface(shape_of<T> shape)
      : base(mv(shape))
  {}

  virtual void stream_data_copy(
    const data_place& dst_memory_node,
    instance_id_t dst_instance_id,
    const data_place& src_memory_node,
    instance_id_t src_instance_id,
    cudaStream_t stream) = 0;

  // Returns prereq
  void data_copy(backend_ctx_untyped& bctx,
                 const data_place& dst_memory_node,
                 instance_id_t dst_instance_id,
                 const data_place& src_memory_node,
                 instance_id_t src_instance_id,
                 event_list& prereqs) override
  {
    cudaStream_t stream;
    auto op = stream_async_op(bctx, dst_memory_node, &stream, prereqs);
    if (bctx.generate_event_symbols())
    {
      op.set_symbol("copy");
    }

    stream_data_copy(dst_memory_node, dst_instance_id, src_memory_node, src_instance_id, stream);

    prereqs = op.end(bctx);
  }
};

/**
 * @brief Simplified data interface for the CUDA streams backend.
 *
 * Implements all primitives of data_interface in terms of to-be-defined
 * primitives that take a stream as a parameter.
 */
template <typename T>
class stream_data_interface_simple : public data_impl_base<T>
{
public:
  using base = data_impl_base<T>;
  using typename base::shape_t;

  stream_data_interface_simple(T p)
      : base(mv(p))
  {}

  stream_data_interface_simple(typename base::shape_t sh)
      : base(sh)
  {}

  // The routine indicates how many bytes have been allocated, or were
  // required if the allocated failed
  virtual void stream_data_allocate(
    backend_ctx_untyped& ctx,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** extra_args,
    cudaStream_t stream) = 0;

  void data_allocate(
    backend_ctx_untyped& ctx,
    block_allocator_untyped& /*unused*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** extra_args,
    event_list& prereqs) final override
  {
    cudaStream_t stream;
    auto op = stream_async_op(ctx, memory_node, &stream, prereqs);
    if (ctx.generate_event_symbols())
    {
      op.set_symbol("alloc");
    }

    stream_data_allocate(ctx, memory_node, instance_id, s, extra_args, stream);

    prereqs = op.end(ctx);
  }

  virtual void stream_data_deallocate(
    backend_ctx_untyped& ctx,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* extra_args,
    cudaStream_t stream) = 0;

  void data_deallocate(
    backend_ctx_untyped& ctx,
    block_allocator_untyped& /*custom_allocator*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* extra_args,
    event_list& prereqs) final override
  {
    cudaStream_t stream;
    auto op = stream_async_op(ctx, memory_node, &stream, prereqs);
    if (ctx.generate_event_symbols())
    {
      op.set_symbol("dealloc ");
    }

    stream_data_deallocate(ctx, memory_node, instance_id, extra_args, stream);

    prereqs = op.end(ctx);
  }

  virtual void stream_data_copy(
    const data_place& dst_memory_node,
    instance_id_t dst_instance_id,
    const data_place& src_memory_node,
    instance_id_t src_instance_id,
    cudaStream_t stream) = 0;

  // Returns prereq
  void data_copy(backend_ctx_untyped& bctx,
                 const data_place& dst_memory_node,
                 instance_id_t dst_instance_id,
                 const data_place& src_memory_node,
                 instance_id_t src_instance_id,
                 event_list& prereqs) override
  {
    cudaStream_t stream;
    auto op = stream_async_op(bctx, dst_memory_node, &stream, prereqs);
    if (bctx.generate_event_symbols())
    {
      op.set_symbol("copy");
    }

    stream_data_copy(dst_memory_node, dst_instance_id, src_memory_node, src_instance_id, stream);

    prereqs = op.end(bctx);
  }
};
} // end namespace cuda::experimental::stf
