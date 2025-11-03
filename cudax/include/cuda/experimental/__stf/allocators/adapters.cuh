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
 * graphs with a large memory footprints. Allocations will be done in the same
 * stream as the one used to launch the graph, and will be destroyed in that
 * stream when the graph has been launched.
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
   * @brief Store the state of the allocator such as the buffers to free
   */
  struct adapter_allocator_state
  {
    adapter_allocator_state(cudaStream_t stream_)
        : stream(stream_)
    {}

    // Resources to release
    ::std::vector<raw_buffer> to_free;

    // stream used to allocate data
    cudaStream_t stream;
  };

  /**
   * @brief allocator interface created within the stream_adapter
   */
  class adapter_allocator : public block_allocator_interface
  {
  public:
    adapter_allocator(::std::shared_ptr<adapter_allocator_state> state_)
        : state(mv(state_))
    {}

    // these are events from a graph_ctx
    void* allocate(
      backend_ctx_untyped&, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& /* prereqs */) override
    {
      // prereqs are unchanged

      void* result;
      EXPECT(!memory_node.is_composite());

      if (memory_node.is_host())
      {
        cuda_safe_call(cudaMallocHost(&result, s));
      }
      else if (memory_node.is_managed())
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

        cuda_safe_call(cudaMallocAsync(&result, s, state->stream));
      }

      return result;
    }

    void deallocate(backend_ctx_untyped&,
                    const data_place& memory_node,
                    event_list& /* prereqs */,
                    void* ptr,
                    [[maybe_unused]] size_t sz) override
    {
      // Do not deallocate buffers, this is done later when we call clear()
      state->to_free.emplace_back(ptr, sz, memory_node);
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
    // To keep track of allocated resources
    ::std::shared_ptr<adapter_allocator_state> state;
  };

public:
  template <typename context_t>
  stream_adapter(context_t& ctx, cudaStream_t stream /*, block_allocator_untyped root_allocator_*/)
      : adapter_state(::std::make_shared<adapter_allocator_state>(stream))
      , alloc(block_allocator<adapter_allocator>(ctx, adapter_state))
  {}

  // Delete copy constructor and copy assignment operator
  stream_adapter(const stream_adapter&)            = delete;
  stream_adapter& operator=(const stream_adapter&) = delete;

  // This is movable, but we don't need to call clear anymore after moving
  stream_adapter(stream_adapter&& other) noexcept
      : adapter_state(other.adapter_state)
      , alloc(other.alloc)
      , cleared_or_moved(other.cleared_or_moved)
  {
    // No need to clear this now that it was moved
    other.cleared_or_moved = true;
  }

  stream_adapter& operator=(stream_adapter&& other) noexcept
  {
    if (this != &other)
    {
      adapter_state    = mv(other.adapter_state);
      alloc            = mv(other.alloc);
      cleared_or_moved = other.cleared_or_moved;

      // Mark the moved-from object as "moved"
      other.cleared_or_moved = true;
    }
    return *this;
  }

  // Destructor
  ~stream_adapter()
  {
    static_assert(::std::is_move_constructible_v<stream_adapter>, "stream_adapter must be move constructible");
    static_assert(::std::is_move_assignable_v<stream_adapter>, "stream_adapter must be move assignable");

    _CCCL_ASSERT(cleared_or_moved, "clear() was not called.");
  }

  /**
   * @brief Free resources allocated by the stream_adapter object
   */
  void clear()
  {
    _CCCL_ASSERT(adapter_state, "Invalid state");
    _CCCL_ASSERT(!cleared_or_moved, "clear() was already called, or the object was moved.");

    // We avoid changing device around every CUDA API call, so we will only
    // change it when necessary, and restore the current device at the end
    // of the loop.
    const int prev_dev_id = cuda_try<cudaGetDevice>();
    int current_dev_id    = prev_dev_id;

    cudaStream_t stream = adapter_state->stream;

    // No need to wait for the stream multiple times
    bool stream_was_synchronized = false;

    for (auto& b : adapter_state->to_free)
    {
      if (b.memory_node.is_host())
      {
        if (!stream_was_synchronized)
        {
          cuda_safe_call(cudaStreamSynchronize(stream));
          stream_was_synchronized = true;
        }
        cuda_safe_call(cudaFreeHost(b.ptr));
      }
      else if (b.memory_node.is_managed())
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

    adapter_state->to_free.clear();

    cleared_or_moved = true;
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
  ::std::shared_ptr<adapter_allocator_state> adapter_state;

  // Note this is using a PIMPL idiom so it's movable
  block_allocator_untyped alloc;

  // We need to call clear() before destroying the object, unless it was moved
  bool cleared_or_moved = false;
};
} // end namespace cuda::experimental::stf
