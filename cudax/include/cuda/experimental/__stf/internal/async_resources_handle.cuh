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
#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh> // for ::std::hash<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>>
#include <cuda/experimental/__stf/utility/stream_to_dev.cuh>
#include <cuda/experimental/__stf/utility/unittest.cuh>

#include <atomic>
#include <mutex>
#include <unordered_map>

#include <cuda.h>

namespace cuda::experimental::stf
{
class green_context_helper;

// Needed to set/get affinity
class exec_place;

/** Sentinel for "no stream" / empty slot. Distinct from any value returned by cuStreamGetId. */
inline constexpr unsigned long long k_no_stream_id = static_cast<unsigned long long>(-1);

/**
 * @brief Returns the unique stream ID from the CUDA driver (cuStreamGetId).
 * @param stream A valid CUDA stream, or nullptr.
 * @return The stream's unique ID, or k_no_stream_id if stream is nullptr.
 */
inline unsigned long long get_stream_id(cudaStream_t stream)
{
  unsigned long long id = 0;
  cuda_safe_call(cuStreamGetId(reinterpret_cast<CUstream>(stream), &id));
  _CCCL_ASSERT(id != k_no_stream_id, "Internal error: cuStreamGetId returned k_no_stream_id");
  return id;
}

/**
 * @brief A class to store a CUDA stream along with metadata
 *
 * It contains
 *  - the stream itself,
 *  - the stream's unique ID from the CUDA driver (cuStreamGetId), or k_no_stream_id if no stream,
 *  - the device index in which the stream resides
 */
struct decorated_stream
{
  decorated_stream() = default;

  decorated_stream(cudaStream_t stream, unsigned long long id, int dev_id = -1)
      : stream(stream)
      , id(id)
      , dev_id(dev_id)
  {}

  /** Construct from stream only; id is from cuStreamGetId, dev_id is -1 (filled lazily when needed). */
  explicit decorated_stream(cudaStream_t stream)
      : stream(stream)
      , id(get_stream_id(stream))
      , dev_id(-1)
  {}

  cudaStream_t stream = nullptr;
  // Unique ID from cuStreamGetId (k_no_stream_id if no stream)
  unsigned long long id = k_no_stream_id;
  // Device in which this stream resides
  int dev_id = -1;
};

class async_resources_handle;

/**
 * @brief A stream_pool object stores a set of streams associated to a specific
 * CUDA context (device, green context, ...)
 *
 * Usage:
 *   pool = get_stream_pool(async_resources, place);
 *   stream = pool.next(place).
 *
 * When a slot is empty, next(place) activates the place (RAII guard) and calls
 * place.create_stream().
 */
struct stream_pool
{
  stream_pool()  = default;
  ~stream_pool() = default;

  /**
   * @brief stream_pool constructor taking a number of slots.
   *
   * Streams are created lazily only via next(place), which activates the place and calls place.create_stream().
   * Slot dev_id and id are set when the stream is created in next().
   */
  explicit stream_pool(size_t n)
      : payload(n, decorated_stream(nullptr, k_no_stream_id, -1))
  {}

  stream_pool(stream_pool&& rhs)
  {
    ::std::lock_guard<::std::mutex> locker(rhs.mtx);
    payload = mv(rhs.payload);
    rhs.payload.clear();
    index = mv(rhs.index);
  }

  /**
   * @brief Get the next stream in the pool; when a slot is empty, activate the place (RAII guard) and call
   * place.create_stream(). Defined in places.cuh so the pool can use exec_place_guard and exec_place::create_stream().
   */
  decorated_stream next(const exec_place& place);

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
  size_t index = 0;
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
    impl()
    {
      const int ndevices = cuda_try<cudaGetDeviceCount>();
      assert(ndevices > 0);
      assert(pool_size > 0);
      assert(data_pool_size > 0);

      per_device_gc_helper.resize(ndevices, nullptr);
      for (auto d : each(ndevices))
      {
        auto& pools = stream_pools[data_place::device(d)];
        stream_pool_init_internal(pools.first, d, pool_size);
        stream_pool_init_internal(pools.second, d, data_pool_size);
      }
    }

    ~impl()
    {
      const int ndevices = static_cast<int>(per_device_gc_helper.size());
      for (auto d : each(ndevices))
      {
        auto& pools = stream_pools[data_place::device(d)];
        stream_pool_cleanup_internal(pools.first);
        stream_pool_cleanup_internal(pools.second);
      }
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
        new_payload.emplace_back(nullptr, k_no_stream_id, dev_id);
      }

      ::std::lock_guard<::std::mutex> locker(p.mtx);
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
        if (e.stream)
        {
          cuda_safe_call(cudaStreamDestroy(e.stream));
        }
      }
    }

  public:
    // This memorize what was the last event used to synchronize a pair of streams
    last_event_per_stream cached_syncs;

    // For each place, a pair of stream_pool objects (computation pool, data transfer pool).
    place_indexed_container<::std::pair<stream_pool, stream_pool>> stream_pools;

    /* Store previously instantiated graphs, indexed by the number of edges and nodes */
    executable_graph_cache cached_graphs;

    ::std::vector<::std::shared_ptr<green_context_helper>> per_device_gc_helper;

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

  auto& stream_pools()
  {
    assert(pimpl);
    return pimpl->stream_pools;
  }

  const auto& stream_pools() const
  {
    assert(pimpl);
    return pimpl->stream_pools;
  }

  bool validate_sync_and_update(unsigned long long dst, unsigned long long src, int event_id)
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

  // Get green context helper with lazy initialization
  ::std::shared_ptr<green_context_helper> get_gc_helper(int dev_id, int sm_count);

  // Register an external green context helper
  void register_gc_helper(int dev_id, ::std::shared_ptr<green_context_helper> helper)
  {
    assert(pimpl);
    assert(dev_id < int(pimpl->per_device_gc_helper.size()));
    pimpl->per_device_gc_helper[dev_id] = ::std::move(helper);
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
