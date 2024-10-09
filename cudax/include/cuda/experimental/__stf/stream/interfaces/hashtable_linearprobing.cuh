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

/**
 * Defines a data interface over the hashtable class, which is based on https://nosferalatu.com/SimpleGPUHashTable.html
 */

#include <cuda/experimental/__stf/internal/hashtable_linearprobing.cuh>
#include <cuda/experimental/__stf/stream/stream_data_interface.cuh>

namespace cuda::experimental::stf
{

class hashtable_stream_interface : public stream_data_interface_simple<hashtable>
{
public:
  using base = stream_data_interface_simple<hashtable>;
  using base::shape_t;

  // Initialize from an existing hashtable
  hashtable_stream_interface(hashtable h)
      : base(mv(h))
  {}

  // Initialize from a shape of hashtable (start empty)
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

    KeyValue* src = src_instance.addr;
    KeyValue* dst = dst_instance.addr;
    size_t sz     = this->shape.get_capacity() * sizeof(KeyValue);

    // NAIVE method !
    cuda_safe_call(cudaMemcpyAsync((void*) dst, (void*) src, sz, kind, s));
  }

  void stream_data_allocate(
    backend_ctx_untyped& /*unused*/,
    const data_place& memory_node,
    instance_id_t instance_id,
    ssize_t& s,
    void** /*unused*/,
    cudaStream_t stream) override
  {
    s                     = this->shape.get_capacity() * sizeof(KeyValue);
    hashtable& local_desc = this->instance(instance_id);

    KeyValue* base_ptr;

    if (memory_node == data_place::host)
    {
      // Fallback to a synchronous method
      cuda_safe_call(cudaStreamSynchronize(stream));
      cuda_safe_call(cudaHostAlloc(&base_ptr, s, cudaHostAllocMapped));
      memset(base_ptr, 0xff, s);
    }
    else
    {
      cuda_safe_call(cudaMallocAsync(&base_ptr, s, stream));

      // We also need to initialize the hashtable
      static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
      cuda_safe_call(cudaMemsetAsync(base_ptr, 0xff, s, stream));
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
    if (memory_node == data_place::host)
    {
      // Fallback to a synchronous method
      cuda_safe_call(cudaStreamSynchronize(stream));
      cuda_safe_call(cudaFreeHost(local_desc.addr));
    }
    else
    {
      cuda_safe_call(cudaFreeAsync(local_desc.addr, stream));
    }
    local_desc.addr = nullptr; // not strictly necessary, but helps debugging
  }
};

template <typename T>
struct streamed_interface_of;

template <>
struct streamed_interface_of<hashtable>
{
  using type = hashtable_stream_interface;
};

} // end namespace cuda::experimental::stf
