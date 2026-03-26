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
 * @brief Implement classes for reusable asynchronous resources such as async_resources_handle
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

#include <cuda/experimental/__stf/internal/exec_affinity.cuh>
#include <cuda/experimental/__stf/internal/executable_graph_cache.cuh>
#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh> // for ::std::hash<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <atomic>
#include <unordered_map>

#include <cuda.h>

namespace cuda::experimental::stf
{
/**
 * @brief A handle which stores resources useful for an efficient asynchronous
 * execution. For example this will store the pools of CUDA streams.
 *
 * This class relies on a PIMPL idiom and can be passed by value. Creating a
 * new object of this type does not initialize any resource, as these will be
 * set lazily.
 */
class async_resources_handle
{
private:
  /**
   * @brief This class implements a matrix to keep track of the previous
   * synchronization that occurred between each pair of streams in our pools.
   *
   * We record the largest event ID used to synchronized between two streams
   * because synchronizing these two streams with an older event (with a lower
   * ID) is implied by the previous synchronization, so it can be skipped thanks
   * to stream-ordering of operations.
   *
   * Keys are pairs of stream IDs from cuStreamGetId.
   */
  class last_event_per_stream
  {
  public:
    // We are trying to insert a dependency from the event with id event_id
    // located on stream "from" to stream "dst" (stream dst waits for the
    // event)
    // Returned value : boolean indicating if we can skip the synchronization
    bool validate_sync_and_update(unsigned long long dst, unsigned long long src, int event_id)
    {
      // If either of the streams has no valid id, do not skip
      if (dst == k_no_stream_id || src == k_no_stream_id)
      {
        return false;
      }

      const auto key = ::std::pair(src, dst);

      if (auto i = interactions.find(key); i != interactions.end())
      {
        // If there is already an entry, potentially update it
        int& last_event_id = i->second;
        if (last_event_id >= event_id)
        {
          // We can skip this synchronization because it was already enforced
          return true;
        }
        last_event_id = event_id;
        return false;
      }
      // These two streams did not interact yet, we need to create a new entry
      interactions[key] = event_id;
      return false;
    }

  private:
    // For each pair of stream IDs (from cuStreamGetId), we keep the last event id
    ::std::unordered_map<::std::pair<unsigned long long, unsigned long long>,
                         int,
                         cuda::experimental::stf::hash<::std::pair<unsigned long long, unsigned long long>>>
      interactions;

    ::std::mutex mtx;
  };

  // We use a pimpl idiom
  class impl
  {
  public:
#if _CCCL_CTK_AT_LEAST(12, 4)
    impl()
    {
      const int ndevices = cuda_try<cudaGetDeviceCount>();
      _CCCL_ASSERT(ndevices > 0, "invalid device count");
      per_device_gc_helper.resize(ndevices, nullptr);
    }
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  public:
    // This memorize what was the last event used to synchronize a pair of streams
    last_event_per_stream cached_syncs;

    /* Store previously instantiated graphs, indexed by the number of edges and nodes */
    executable_graph_cache cached_graphs;

#if _CCCL_CTK_AT_LEAST(12, 4)
    ::std::vector<::std::shared_ptr<green_context_helper>> per_device_gc_helper;
#endif // _CCCL_CTK_AT_LEAST(12, 4)

    mutable exec_affinity affinity;
  };

  mutable ::std::shared_ptr<impl> pimpl;

public:
  async_resources_handle()
      : pimpl(::std::make_shared<impl>())
  {}

  explicit async_resources_handle(::std::nullptr_t) {}

  explicit operator bool() const
  {
    return pimpl != nullptr;
  }

  bool validate_sync_and_update(unsigned long long dst, unsigned long long src, int event_id)
  {
    assert(pimpl);
    return pimpl->cached_syncs.validate_sync_and_update(dst, src, event_id);
  }

  // The graph is only used during the call (to update or instantiate); it is never stored, so the
  // caller only needs to keep it valid for the duration of the call.
  ::cuda::std::pair<::std::shared_ptr<cudaGraphExec_t>, bool>
  cached_graphs_query(size_t nnodes, size_t nedges, cudaGraph_t g)
  {
    _CCCL_ASSERT(pimpl, "async_resources_handle is not initialized");
    return pimpl->cached_graphs.query(nnodes, nedges, g);
  }

  ::cuda::std::pair<::std::shared_ptr<cudaGraphExec_t>, bool> cached_graphs_query(cudaGraph_t g)
  {
    size_t nedges;
    size_t nnodes;

    cuda_safe_call(cudaGraphGetNodes(g, nullptr, &nnodes));
#if _CCCL_CTK_AT_LEAST(13, 0)
    cuda_safe_call(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &nedges));
#else // _CCCL_CTK_AT_LEAST(13, 0)
    cuda_safe_call(cudaGraphGetEdges(g, nullptr, nullptr, &nedges));
#endif // _CCCL_CTK_AT_LEAST(13, 0)

    _CCCL_ASSERT(pimpl, "async_resources_handle is not initialized");
    return cached_graphs_query(nnodes, nedges, g);
  }

#if _CCCL_CTK_AT_LEAST(12, 4)
  // Get the green context helper cached for this device (or let the user initialize it)
  auto& gc_helper(int dev_id)
  {
    assert(pimpl);
    assert(dev_id < int(pimpl->per_device_gc_helper.size()));
    return pimpl->per_device_gc_helper[dev_id];
  }

  // Get green context helper with lazy initialization
  ::std::shared_ptr<green_context_helper> get_gc_helper(int dev_id, int sm_count)
  {
    assert(pimpl);
    assert(dev_id < int(pimpl->per_device_gc_helper.size()));
    auto& h = pimpl->per_device_gc_helper[dev_id];
    if (!h)
    {
      h = ::std::make_shared<green_context_helper>(sm_count, dev_id);
    }
    return h;
  }

  // Register an external green context helper
  void register_gc_helper(int dev_id, ::std::shared_ptr<green_context_helper> helper)
  {
    assert(pimpl);
    assert(dev_id < int(pimpl->per_device_gc_helper.size()));
    pimpl->per_device_gc_helper[dev_id] = ::std::move(helper);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 4)

  exec_affinity& get_affinity()
  {
    assert(pimpl);
    return pimpl->affinity;
  }

  const exec_affinity& get_affinity() const
  {
    assert(pimpl);
    return pimpl->affinity;
  }

  bool has_affinity() const
  {
    assert(pimpl);
    return pimpl->affinity.has_affinity();
  }

  void push_affinity(::std::vector<::std::shared_ptr<exec_place>> p) const
  {
    assert(pimpl);
    pimpl->affinity.push(mv(p));
  }

  void push_affinity(::std::shared_ptr<exec_place> p) const
  {
    assert(pimpl);
    pimpl->affinity.push(mv(p));
  }

  void pop_affinity() const
  {
    assert(pimpl);
    pimpl->affinity.pop();
  }

  const ::std::vector<::std::shared_ptr<exec_place>>& current_affinity() const
  {
    assert(pimpl);
    return pimpl->affinity.top();
  }
};

#ifdef UNITTESTED_FILE
/*
 * This test ensures that the async_resources_handle type is default
 * constructible, which for example makes it possible to use it in a raft
 * handle.
 */
UNITTEST("async_resources_handle is_default_constructible")
{
  static_assert(::std::is_default_constructible<async_resources_handle>::value,
                "async_resources_handle must be default constructible");
};
#endif
} // namespace cuda::experimental::stf
