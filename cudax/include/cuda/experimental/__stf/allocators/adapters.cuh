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
 * @brief Methods to adapt allocators to rely on a third-party allocator
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

#include <cuda/experimental/__stf/allocators/block_allocator.cuh>

namespace cuda::experimental::stf
{

/**
 * @brief Allocator which defers allocations to the CUDA asynchronous memory
 * allocations APIs (cudaMallocAsync, cudaFreeAsync)
 *
 * This can be used as an alternative in CUDA graphs to avoid creating CUDA
 * graphs with a large memory footprints.
 */
class stream_adapter
{
  /**
   * @brief Description of an allocated buffer
   */
  struct raw_buffer
  {
    raw_buffer(void* ptr_, size_t sz_, data_place memory_node_)
        : ptr(ptr_)
        , sz(sz_)
        , memory_node(memory_node_)
    {}

    void* ptr;
    size_t sz;
    data_place memory_node;
  };

  /**
   * @brief allocator interface created within the stream_adapter
   */
  class adapter_allocator : public block_allocator_interface
  {
  public:
    adapter_allocator(cudaStream_t stream_, stream_adapter* sa_)
        : stream(stream_)
        , sa(sa_)
    {}

    // these are events from a graph_ctx
    void* allocate(
      backend_ctx_untyped&, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& /* prereqs */) override
    {
      // prereqs are unchanged

      void* result;
      EXPECT(!memory_node.is_composite());

      if (memory_node == data_place::host)
      {
        cuda_safe_call(cudaMallocHost(&result, s));
      }
      else if (memory_node == data_place::managed)
      {
        cuda_safe_call(cudaMallocManaged(&result, s));
      }
      else
      {
        const int prev_dev_id = cuda_try<cudaGetDevice>();
        // (Note device_ordinal works with green contexts as well)
        const int target_dev_id = device_ordinal(memory_node);

        if (memory_node.is_green_ctx())
        {
          fprintf(stderr,
                  "Pretend we use cudaMallocAsync on green context (using device %d in reality)\n",
                  device_ordinal(memory_node));
        }

        if (prev_dev_id != target_dev_id)
        {
          cuda_safe_call(cudaSetDevice(target_dev_id));
        }

        SCOPE(exit)
        {
          if (target_dev_id != prev_dev_id)
          {
            cuda_safe_call(cudaSetDevice(prev_dev_id));
          }
        };

        cuda_safe_call(cudaMallocAsync(&result, s, stream));
      }

      return result;
    }

    void deallocate(
      backend_ctx_untyped&, const data_place& memory_node, event_list& /* prereqs */, void* ptr, size_t sz) override
    {
      // Do not deallocate buffers, this is done later after
      sa->to_free.emplace_back(ptr, sz, memory_node);
      // Prereqs are unchanged
    }

    event_list deinit(backend_ctx_untyped&) override
    {
      // no op
      return event_list();
    }

    ::std::string to_string() const override
    {
      return "stream_adapter";
    }

  private:
    cudaStream_t stream;
    stream_adapter* sa;
  };

public:
  template <typename context_t>
  stream_adapter(context_t& ctx, cudaStream_t stream_ /*, block_allocator_untyped root_allocator_*/)
      : stream(stream_) /*, root_allocator(mv(root_allocator_))*/
  {
    alloc = block_allocator<adapter_allocator>(ctx, stream, this);
  }

  /**
   * @brief Free resources allocated by the stream_adapter object
   */
  void clear()
  {
    // We avoid changing device around every CUDA API call, so we will only
    // change it when necessary, and restore the current device at the end
    // of the loop.
    const int prev_dev_id = cuda_try<cudaGetDevice>();
    int current_dev_id    = prev_dev_id;

    // No need to wait for the stream multiple times
    bool stream_was_synchronized = false;

    for (auto& b : to_free)
    {
      if (b.memory_node == data_place::host)
      {
        if (!stream_was_synchronized)
        {
          cuda_safe_call(cudaStreamSynchronize(stream));
          stream_was_synchronized = true;
        }
        cuda_safe_call(cudaFreeHost(b.ptr));
      }
      else if (b.memory_node == data_place::managed)
      {
        if (!stream_was_synchronized)
        {
          cuda_safe_call(cudaStreamSynchronize(stream));
          stream_was_synchronized = true;
        }
        cuda_safe_call(cudaFree(b.ptr));
      }
      else
      {
        // (Note device_ordinal works with green contexts as well)
        int target_dev_id = device_ordinal(b.memory_node);
        if (current_dev_id != target_dev_id)
        {
          cuda_safe_call(cudaSetDevice(target_dev_id));
          current_dev_id = target_dev_id;
        }

        cuda_safe_call(cudaFreeAsync(b.ptr, stream));
      }
    }

    if (current_dev_id != prev_dev_id)
    {
      cuda_safe_call(cudaSetDevice(prev_dev_id));
    }

    to_free.clear();
  }

  /**
   * @brief Get the underlying block allocator so that we can set the
   * allocator of a context, or a logical data
   */
  auto& allocator()
  {
    return alloc;
  }

private:
  cudaStream_t stream;

  block_allocator_untyped alloc;

  ::std::vector<raw_buffer> to_free;
};

} // end namespace cuda::experimental::stf
