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
      // prereqs are unchanged - use raw allocation with the adapter's stream
      EXPECT(!memory_node.is_composite());
      return memory_node.allocate(s, state->stream);
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

    const cudaStream_t stream = adapter_state->stream;

    // Deallocate buffers one at a time, popping from the back. If any CUDA
    // call below throws, ``to_free`` still holds the un-deallocated entries
    // and ``cleared_or_moved`` stays false, so the caller can recover (catch
    // and retry, or let the destructor's assertion fire with accurate state).
    // Order across buffers does not matter because each ``raw_buffer`` is
    // independent.
    //
    // Subtlety: we do not call ``cuda_try(cudaStreamSynchronize(...))`` here
    // because we want the just-popped buffer's ``deallocate`` to run even on
    // sync failure -- losing the descriptor without freeing would leak. We
    // capture the sync status, do the deallocation, then surface the sync
    // error via ``cuda_try(cudaStreamSynchronize_result)`` afterwards. We
    // deliberately do not wrap that in a SCOPE guard: ``data_place_*::
    // deallocate`` itself can throw (it uses ``cuda_try`` internally for
    // ``cudaFreeHost`` / ``cudaFree`` / ``cudaFreeAsync``), and SCOPE bodies
    // are ``noexcept``, so a deallocate-throw during unwinding would call
    // ``std::terminate``.
    bool cudaStreamSynchronize_was_called = false;
    while (!adapter_state->to_free.empty())
    {
      const auto b = mv(adapter_state->to_free.back());
      adapter_state->to_free.pop_back();

      cudaError_t cudaStreamSynchronize_result = cudaSuccess;
      if (!cudaStreamSynchronize_was_called && !b.memory_node.allocation_is_stream_ordered())
      {
        cudaStreamSynchronize_result     = cudaStreamSynchronize(stream);
        cudaStreamSynchronize_was_called = true;
      }

      // The following two lines may throw, in which case we're left in steady state
      b.memory_node.deallocate(b.ptr, b.sz, stream);
      cuda_try(cudaStreamSynchronize_result);
    }

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
