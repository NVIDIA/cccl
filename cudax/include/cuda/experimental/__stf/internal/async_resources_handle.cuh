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
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh> // for ::std::hash<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>>
#include <cuda/experimental/__stf/utility/stream_to_dev.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace cuda::experimental::stf
{

class green_context_helper;

// Needed to set/get affinity
class exec_place;

/**
 * @brief A class to store a CUDA stream along with a few information to avoid CUDA queries
 *
 * It contains
 *  - the stream itself,
 *  - a unique id (proper to CUDASTF, and only valid for streams in our pool, or equal to -1),
 *  - the pool associated to the unique ID, when valid
 *  - the device index in which the stream is
 */
struct decorated_stream
{
  decorated_stream(cudaStream_t stream = nullptr, ::std::ptrdiff_t id = -1, int dev_id = -1)
      : stream(stream)
      , id(id)
      , dev_id(dev_id)
  {}

  cudaStream_t stream = nullptr;
  // Unique ID (-1 if this is not part of our pool)
  ::std::ptrdiff_t id = -1;
  // Device in which this stream resides
  int dev_id = -1;
};

class async_resources_handle;

/**
 * @brief A stream_pool object stores a set of streams associated to a specific CUDA context (device, green context,
 * ...)
 *
 * Stream pools are populated lazily.
 */
struct stream_pool
{
  stream_pool()  = default;
  ~stream_pool() = default;

  /**
   * @brief stream_pool constructor taking a number of slots, an optional CUDA context and a device id.
   *
   * Streams are initialized lazily when calling next().
   */
  stream_pool(size_t n, int dev_id, CUcontext cuctx = nullptr)
      : payload(n, decorated_stream(nullptr, -1, dev_id))
      , dev_id(dev_id)
      , primary_ctx(cuctx)
  {}

  stream_pool(stream_pool&& rhs)
  {
    ::std::lock_guard<::std::mutex> locker(rhs.mtx);
    // ATTENTION: add to these whenever adding members
    payload = mv(rhs.payload);
    rhs.payload.clear();
    index  = mv(rhs.index);
    dev_id = mv(rhs.dev_id);
  }

  /**
   * @brief Get the next stream in the pool
   *
   * The stream is wrapped into a decorated_stream class to embed information
   * such as its position in the pool, or the device id.
   */
  decorated_stream next()
  {
    ::std::lock_guard<::std::mutex> locker(mtx);
    assert(index < payload.size());

    auto& result = payload.at(index);

    assert(result.dev_id != -1);

    /* Note that we do not check if id is initialized to a unique id (!=
     * -1) because we could create pools of unregistered streams. Such
     * identifiers are useful to avoid some synchronizations, but are not
     * mandatory. */

    /* Streams are created lazily, so there may be no stream yet */
    if (!result.stream)
    {
      if (primary_ctx)
      {
        cuda_safe_call(cuCtxPushCurrent(primary_ctx));

        /* Create a CUDA stream on that context */
        cuda_safe_call(cudaStreamCreate(&result.stream));

        CUcontext dummy;
        cuda_safe_call(cuCtxPopCurrent(&dummy));
      }
      else
      {
        const auto old_dev_id = cuda_try<cudaGetDevice>();
        if (old_dev_id != dev_id)
        {
          cuda_safe_call(cudaSetDevice(dev_id));
        }

        /* Create a CUDA stream on that device */
        cuda_safe_call(cudaStreamCreate(&result.stream));

        if (old_dev_id != dev_id)
        {
          cuda_safe_call(cudaSetDevice(old_dev_id));
        }
      }
    }

    if (++index >= payload.size())
    {
      index = 0;
    }

    return result;
  }

  // To iterate over all entries of the pool
  using iterator = ::std::vector<decorated_stream>::iterator;
  iterator begin()
  {
    return payload.begin();
  }
  iterator end()
  {
    return payload.end();
  }

  /**
   * @brief Number of streams in the pool
   *
   * CUDA streams are initialized lazily, so this gives the number of slots
   * available in the pool, not the number of streams initialized.
   */
  size_t size() const
  {
    ::std::lock_guard<::std::mutex> locker(mtx);
    return payload.size();
  }

  // ATTENTION: see move ctor
  mutable ::std::mutex mtx;
  ::std::vector<decorated_stream> payload;
  size_t index          = 0;
  int dev_id            = 1;
  CUcontext primary_ctx = nullptr;
};

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
  // TODO: optimize based on measurements

public:
  static constexpr size_t pool_size      = 4;
  static constexpr size_t data_pool_size = 4;

private:
  /**
   * @brief A helper class to maintain a set of available IDs, and attributes IDs
   */
  class id_pool
  {
  public:
    ~id_pool()
    {
      assert(released.load() == current.load());
    }

    ::std::ptrdiff_t get_unique_id(size_t cnt = 1)
    {
      // Use fetch_add to atomically increment current and return the previous value
      return current.fetch_add(cnt);
    }

    void release_unique_id(::std::ptrdiff_t /* id */, size_t cnt = 1)
    {
      // Use fetch_add to atomically increment released
      released.fetch_add(cnt);
    }

  private:
    // next available ID
    ::std::atomic<::std::ptrdiff_t> current{0};
    // Number of IDs released, for bookkeeping
    ::std::atomic<::std::ptrdiff_t> released{0};
  };

  /**
   * @brief This class implements a matrix to keep track of the previous
   * synchronization that occurred between each pair of streams in our pools.
   *
   * We record the largest event ID used to synchronized between two streams
   * because synchronizing these two streams with an older event (with a lower
   * ID) is implied by the previous synchronization, so it can be skipped thanks
   * to stream-ordering of operations.
   *
   * This is implemented as a hash table where keys are pairs of IDs.
   */
  class last_event_per_stream
  {
  public:
    // We are trying to insert a dependency from the event with id event_id
    // located on stream "from" to stream "dst" (stream dst waits for the
    // event)
    // Returned value : boolean indicating if we can skip the synchronization
    bool validate_sync_and_update(::std::ptrdiff_t dst, ::std::ptrdiff_t src, int event_id)
    {
      // If either of the streams is not from the pool, do not skip
      if (dst == -1 || src == -1)
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
    // For each pair of unique IDs, we keep the last event id
    ::std::unordered_map<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>,
                         int,
                         cuda::experimental::stf::hash<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>>>
      interactions;

    ::std::mutex mtx;
  };

  // We use a pimpl idiom
  class impl
  {
  public:
    impl()
    {
      // Save current device so that this goes unnoticed
      const int ndevices = cuda_try<cudaGetDeviceCount>();
      assert(ndevices > 0);
      assert(pool_size > 0);
      assert(data_pool_size > 0);

      pool.resize(ndevices);
      per_device_gc_helper.resize(ndevices, nullptr);
      /* For every device, we keep two pools, one dedicated to computation,
       * the other for auxiliary methods such as data transfers. This is intended to
       * improve overlapping of transfers and computation, for example. */
      for (auto d : each(ndevices))
      {
        stream_pool_init_internal(pool[d].first, d, pool_size);
        stream_pool_init_internal(pool[d].second, d, data_pool_size);
      }
    }

    ~impl()
    {
      // Release the resources for each device (for each device, there is a
      // stream_pool class to store CUDA streams for computation and another
      // for data transfers)
      for (auto& p : pool)
      {
        stream_pool_cleanup_internal(p.first);
        stream_pool_cleanup_internal(p.second);
      }
    }

    stream_pool& get_device_stream_pool(int dev_id, bool for_computation)
    {
      assert(dev_id < int(pool.size()));
      return for_computation ? pool[dev_id].first : pool[dev_id].second;
    }

  private:
    // Initialize a stream pool for the dev_id CUDA device with n slots which will be populated lazily
    void stream_pool_init_internal(stream_pool& p, int dev_id, size_t n)
    {
      assert(n > 0);

      // Do most of the work outside the critical section
      ::std::vector<decorated_stream> new_payload;
      new_payload.reserve(n);
      for (auto i : each(n))
      {
        ::std::ignore = i;
        new_payload.emplace_back(nullptr, ids.get_unique_id(), dev_id);
      }

      ::std::lock_guard<::std::mutex> locker(p.mtx);
      p.dev_id  = dev_id;
      p.payload = std::move(new_payload);
    }

    void stream_pool_cleanup_internal(stream_pool& p)
    {
      ::std::vector<decorated_stream> goner;
      {
        ::std::lock_guard<::std::mutex> locker(p.mtx);
        p.payload.swap(goner);
      }
      // Clean up outside the critical section
      for (auto& e : goner)
      {
        ids.release_unique_id(e.id);
        if (e.stream)
        {
          cuda_safe_call(cudaStreamDestroy(e.stream));
        }
      }
    }

  public:
    // These are constructed and destroyed in reversed order
    id_pool ids;

    // This memorize what was the last event used to synchronize a pair of streams
    last_event_per_stream cached_syncs;

    // For each device, a pair of stream_pool objects, each stream_pool objects
    // stores a pool of streams on this device
    ::std::vector<::std::pair<stream_pool, stream_pool>> pool;

    /* Store previously instantiated graphs, indexed by the number of edges and nodes */
    executable_graph_cache cached_graphs;

    ::std::vector<::std::shared_ptr<green_context_helper>> per_device_gc_helper;

    mutable exec_affinity affinity;
  };

  ::std::shared_ptr<impl> pimpl;

public:
  async_resources_handle()
      : pimpl(::std::make_shared<impl>())
  {}

  explicit async_resources_handle(::std::nullptr_t) {}

  explicit operator bool() const
  {
    return pimpl != nullptr;
  }

  stream_pool& get_device_stream_pool(int dev_id, bool for_computation)
  {
    assert(pimpl);
    return pimpl->get_device_stream_pool(dev_id, for_computation);
  }

  ::std::ptrdiff_t get_unique_id(size_t cnt = 1)
  {
    assert(pimpl);
    return pimpl->ids.get_unique_id(cnt);
  }

  void release_unique_id(::std::ptrdiff_t id, size_t cnt = 1)
  {
    assert(pimpl);
    return pimpl->ids.release_unique_id(id, cnt);
  }

  bool validate_sync_and_update(::std::ptrdiff_t dst, ::std::ptrdiff_t src, int event_id)
  {
    assert(pimpl);
    return pimpl->cached_syncs.validate_sync_and_update(dst, src, event_id);
  }

  ::cuda::std::pair<::std::shared_ptr<cudaGraphExec_t>, bool>
  cached_graphs_query(size_t nnodes, size_t nedges, ::std::shared_ptr<cudaGraph_t> g)
  {
    assert(pimpl);
    return pimpl->cached_graphs.query(nnodes, nedges, mv(g));
  }

  // Get the green context helper cached for this device (or let the user initialize it)
  auto& gc_helper(int dev_id)
  {
    assert(pimpl);
    assert(dev_id < int(pimpl->per_device_gc_helper.size()));
    return pimpl->per_device_gc_helper[dev_id];
  }

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

//! @brief Registers a user-provided CUDA stream with asynchronous resources
//!
//! @details This optimization records a CUDA stream in the provided asynchronous resources handle,
//! creating a decorated_stream object that encapsulates:
//! - The original stream handle
//! - A unique identifier for stream tracking
//! - The associated device ID
//!
//! @param[in,out] async_resources Handle to asynchronous resources manager
//! @param[in] user_stream Raw CUDA stream to register. Must be a valid stream.
//!
//! @return decorated_stream Object containing:
//!         - Original stream handle
//!         - Unique ID from async_resources
//!         - Device ID associated with the stream
//!
//! @pre `user_stream` must be a valid CUDA stream created with `cudaStreamCreate` or equivalent
//! @note This registration is an optimization to avoid repeated stream metadata lookups
//!       in performance-critical code paths
inline decorated_stream register_stream(async_resources_handle& async_resources, cudaStream_t user_stream)
{
  // Get a unique ID
  const auto id    = async_resources.get_unique_id();
  const int dev_id = get_device_from_stream(user_stream);

  return decorated_stream(user_stream, id, dev_id);
}

//! @brief Unregisters a decorated CUDA stream from asynchronous resources
//!
//! @details Performs cleanup operations to release resources associated with a previously
//! registered stream. This includes:
//! - Releasing the unique ID back to the resource manager
//! - Invalidating the decorated stream's internal ID
//!
//! @param[in,out] async_resources Handle to asynchronous resources manager
//! @param[in,out] dstream Decorated stream to unregister. Its `id` will be set to -1.
//!
//! @pre `dstream.id` must be valid (≥ 0) before calling this function
//! @post `dstream.id == -1` and associated resources are released
//! @note Should be paired with register_stream() for proper resource management
inline void unregister_stream(async_resources_handle& async_resources, decorated_stream& dstream)
{
  async_resources.release_unique_id(dstream.id);
  // reset the decorated stream
  dstream.id = -1;
}

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
