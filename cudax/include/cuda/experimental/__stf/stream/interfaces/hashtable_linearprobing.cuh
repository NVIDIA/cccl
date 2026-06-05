//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file
 *
 * @brief Defines a data interface over the hashtable class, which is based on
 * https://nosferalatu.com/SimpleGPUHashTable.html
 */

#include <cuda/experimental/__stf/internal/hashtable_linearprobing.cuh>
#include <cuda/experimental/__stf/stream/stream_data_interface.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief A simple example of a hashtable data interface that can be used as a logical data in the ``stream_ctx``.
 */
class hashtable_stream_interface : public stream_data_interface_simple<hashtable>
{
public:
  using base = stream_data_interface_simple<hashtable>;
  using base::shape_t;

  /**
   * @brief Initialize from an existing hashtable
   */
  hashtable_stream_interface(hashtable h)
      : base(mv(h))
  {}

  /**
   * @brief Initialize from a shape of hashtable (start empty)
   * @overload
   */
  hashtable_stream_interface(shape_t /*unused*/)
      : base(hashtable(nullptr))
  {}

  void stream_data_copy(
    const data_place& dst_memory_node,
    instance_id_t dst_instance_id,
    const data_place& src_memory_node,
    instance_id_t src_instance_id,
    cudaStream_t s) override
  {
    ::std::ignore = src_memory_node;
    ::std::ignore = dst_memory_node;
    assert(src_memory_node != dst_memory_node);

    cudaMemcpyKind kind = cudaMemcpyDefault;

    const hashtable& src_instance = this->instance(src_instance_id);
    const hashtable& dst_instance = this->instance(dst_instance_id);

    reserved::KeyValue* src = src_instance.addr;
    reserved::KeyValue* dst = dst_instance.addr;
    size_t sz               = this->shape.get_capacity() * sizeof(reserved::KeyValue);

    // NAIVE method !
    // cudaMemcpyAsync is an overload set (cuda_runtime.h alternate-spelling wrapper),
    // so it keeps the runtime-status cuda_try form.
    cuda_try(cudaMemcpyAsync((void*) dst, (void*) src, sz, kind, s));
  }

  void stream_data_allocate(
    backend_ctx_untyped& /*unused*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    ::std::ptrdiff_t& s,
    void** /*unused*/,
    cudaStream_t stream) override
  {
    s                     = this->shape.get_capacity() * sizeof(reserved::KeyValue);
    hashtable& local_desc = this->instance(instance_id);

    reserved::KeyValue* base_ptr;

    if (memory_node.is_host())
    {
      // Fallback to a synchronous method. cudaHostAlloc is an overload set
      // (cuda_runtime.h templated wrapper), so it keeps the runtime-status form.
      cuda_try<cudaStreamSynchronize>(stream);
      cuda_try(cudaHostAlloc(&base_ptr, s, cudaHostAllocMapped));
      memset(base_ptr, 0xff, s);
    }
    else
    {
      // cudaMallocAsync is an overload set (templated wrapper), so it keeps the
      // runtime-status form.
      cuda_try(cudaMallocAsync(&base_ptr, s, stream));
      // Free the buffer if the initialization below throws.
      SCOPE(fail)
      {
        cuda_safe_call(cudaFreeAsync(base_ptr, stream));
      };

      // We also need to initialize the hashtable
      static_assert(reserved::kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
      cuda_try<cudaMemsetAsync>(base_ptr, 0xff, s, stream);
    }

    local_desc.addr = base_ptr;
  }

  void stream_data_deallocate(
    backend_ctx_untyped& /*unused*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    void* /*unused*/,
    cudaStream_t stream) override
  {
    hashtable& local_desc = this->instance(instance_id);
    if (memory_node.is_host())
    {
      // Fallback to a synchronous method
      cuda_try<cudaStreamSynchronize>(stream);
      cuda_try<cudaFreeHost>(local_desc.addr);
    }
    else
    {
      cuda_try<cudaFreeAsync>(local_desc.addr, stream);
    }
    local_desc.addr = nullptr; // not strictly necessary, but helps debugging
  }
};

/// @cond
// Forward declaration
template <typename T>
struct streamed_interface_of;
/// @endcond

template <>
struct streamed_interface_of<hashtable>
{
  using type = hashtable_stream_interface;
};
} // end namespace cuda::experimental::stf
