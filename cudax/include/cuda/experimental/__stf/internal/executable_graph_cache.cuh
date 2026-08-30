//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Cache mechanism to reuse or update executable CUDA graphs
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

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/hash.cuh> // for ::std::hash<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>>
#include <cuda/experimental/__stf/utility/pretty_print.cuh>
#include <cuda/experimental/__stf/utility/source_location.cuh>

#include <mutex>
#include <unordered_map>

namespace cuda::experimental::stf
{
namespace reserved
{
// This tries to instantiate the graph by updating an existing executable graph
// the returned value indicates whether the update was successful or not
inline bool try_updating_executable_graph(cudaGraphExec_t exec_graph, cudaGraph_t graph)
{
  cudaGraphExecUpdateResultInfo resultInfo;
  cudaGraphExecUpdate(exec_graph, graph, &resultInfo);

  // Be sure to "erase" the last error
  cudaError_t res = cudaGetLastError();

  return (res == cudaSuccess);
}

// Instantiate a CUDA graph
inline ::std::shared_ptr<cudaGraphExec_t> graph_instantiate(cudaGraph_t g)
{
  // The handle stays null if instantiation throws below: the deleter must
  // not destroy it in that case, or the abort would mask the real error.
  ::std::shared_ptr<cudaGraphExec_t> res{new cudaGraphExec_t{}, [](cudaGraphExec_t* p) {
                                           if (*p)
                                           {
                                             cuda_safe_call(cudaGraphExecDestroy(*p));
                                           }
                                           delete p;
                                         }};

  // Automatically free graph-owned async allocations between launches. This
  // lets graphs containing cudaMallocAsync / cudaMemAllocNode allocations be
  // relaunched even when the corresponding free is outside the captured graph.
  *res = cuda_try<cudaGraphInstantiateWithFlags>(g, cudaGraphInstantiateFlagAutoFreeOnLaunch);

  return res;
}
} // end namespace reserved

// To get information about how it was used
class executable_graph_cache_stat
{
public:
  size_t instantiate_cnt = 0;
  size_t update_cnt      = 0;
  size_t nnodes          = 0;
  size_t nedges          = 0;

  executable_graph_cache_stat& operator+=(const executable_graph_cache_stat& other)
  {
    instantiate_cnt += other.instantiate_cnt;
    update_cnt += other.update_cnt;
    nnodes += other.nnodes;
    nedges += other.nedges;
    return *this;
  }
};

class executable_graph_cache
{
public:
  executable_graph_cache()
  {
    cache_size_limit = 512 * 1024 * 1024;

    // Maximum size of the executable graph cache (in MB) per device
    // Cache is disabled if the size is 0
    const char* str = getenv("CUDASTF_GRAPH_CACHE_SIZE_MB");
    if (str)
    {
      cache_size_limit = atol(str) * 1024 * 1024;
    }

    const int ndevices = cuda_try<cudaGetDeviceCount>();

    // One individual cache per device (TODO per execution place at some point
    // if we consider green contexts or multi-gpu graphs ?)
    cached_graphs.resize(ndevices);

    // Initialize the footprint per device too
    total_cache_footprint.resize(ndevices, 0);
  }

  // One entry of the cache
  struct entry
  {
    entry(executable_graph_cache* cache,
          ::std::shared_ptr<cudaGraphExec_t> exec_g_,
          cudaStream_t stream_,
          unsigned long long stream_id_,
          size_t footprint)
        : cache(cache)
        , exec_g(mv(exec_g_))
        , stream(stream_)
        , stream_id(stream_id_)
        , footprint(footprint)
    {
      last_use = cache->index++;
    }

    // Update the last_use field to mark that this entry was used recently
    void lru_refresh()
    {
      last_use = cache->index++;
    }

    executable_graph_cache* cache;
    ::std::shared_ptr<cudaGraphExec_t> exec_g;
    // The binding identity is the driver-assigned stream id, which is unique
    // for the lifetime of the process: a cudaStream_t handle value can be
    // recycled after cudaStreamDestroy, so comparing handles could falsely
    // match an entry bound to a dead stream against an unrelated new one.
    // The raw handle is kept only to probe idleness, which is meaningful
    // only while the bound stream is alive (see query_stream_state).
    cudaStream_t stream;
    unsigned long long stream_id;
    size_t last_use;
    size_t footprint;
  };

  // TODO we should not have to redefine this one again
  struct hash_pair
  {
    size_t operator()(const std::pair<size_t, size_t>& p) const
    {
      auto h1 = ::std::hash<size_t>{}(p.first); // Hash the first element
      auto h2 = ::std::hash<size_t>{}(p.second); // Hash the second element
      return h1 ^ (h2 << 1); // Combine the two hash values
    }
  };

  // On each device, we have a map indexed by pairs of edge/vertex count
  using per_device_map_t = ::std::unordered_multimap<::std::pair<size_t, size_t>, entry, hash_pair>;

  // Check if there is a matching entry (and update it if necessary)
  // the returned bool indicate is this is a cache hit (true = cache hit, false = cache miss)
  // The graph g is only used during this call (for update or instantiate); it is never stored.
  ::cuda::std::pair<::std::shared_ptr<cudaGraphExec_t>, bool>
  query(size_t nnodes, size_t nedges, cudaGraph_t g, cudaStream_t stream)
  {
    ::std::lock_guard<::std::mutex> guard(mutex);

    int dev_id = cuda_try<cudaGetDevice>();
    _CCCL_ASSERT(dev_id < int(cached_graphs.size()), "invalid device id value");

    const unsigned long long stream_id = stream_unique_id(stream);

    auto range = cached_graphs[dev_id].equal_range({nnodes, nedges});
    for (auto it = range.first; it != range.second; ++it)
    {
      auto& e = it->second;
      // Executable graphs are only reused on the stream to which the cache
      // entry is bound. In addition to preventing CUDA from serializing
      // concurrent launches of one executable on different streams, this
      // gives us an explicit completion check before the host-side update.
      // The caller's stream is alive by definition, so probing it is safe;
      // a caller stream in capture reads as busy (a query would invalidate
      // the capture), falling through to a fresh instantiation.
      if (e.stream_id != stream_id || query_stream_state(stream) != stream_state::idle)
      {
        continue;
      }

      if (reserved::try_updating_executable_graph(*e.exec_g, g))
      {
        // update the last use index for the LRU algorithm
        e.lru_refresh();

        // We have successfully updated the graph, this is a cache hit
        return ::cuda::std::make_pair(e.exec_g, true);
      }
    }

    // There was no match, so we ensure we have enough memory (or reclaim
    // some), and then instantiate a new graph and put it in the cache.

    // Rough footprint estimate of the graph based on the number of nodes (this
    // is really an approximation)
    size_t footprint = nnodes * 10240;
    if (total_cache_footprint[dev_id] + footprint > cache_size_limit)
    {
      reclaim(dev_id, total_cache_footprint[dev_id] + footprint - cache_size_limit);
    }

    auto exec_g = reserved::graph_instantiate(g);

    // If we maintain a cache, store the executable graph
    if (cache_size_limit != 0)
    {
      cached_graphs[dev_id].insert(
        {::std::make_pair(nnodes, nedges), entry(this, exec_g, stream, stream_id, footprint)});
      total_cache_footprint[dev_id] += footprint;
    }

    return ::cuda::std::make_pair(exec_g, false);
  }

private:
  // The driver-assigned stream id: unique for the process lifetime, unlike
  // the handle value (see entry::stream_id).
  static unsigned long long stream_unique_id(cudaStream_t stream)
  {
    unsigned long long id = 0;
    cuda_safe_call(cudaStreamGetId(stream, &id));
    return id;
  }

  enum class stream_state
  {
    idle,
    busy,
    unavailable
  };

  // Probe a stream without ever throwing and without touching a capture:
  // cudaStreamQuery on a capturing stream would invalidate that capture (a
  // cross-thread hazard when reclaim probes another context's stream), so
  // capture status is checked first with the capture-legal API. Errors from
  // either call (e.g. a destroyed handle for an entry whose bound stream the
  // cache does not own) read as `unavailable`: such an entry is neither
  // reusable nor provably safe to destroy.
  static stream_state query_stream_state(cudaStream_t stream)
  {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess)
    {
      cudaGetLastError();
      return stream_state::unavailable;
    }
    if (capture != cudaStreamCaptureStatusNone)
    {
      return stream_state::busy;
    }
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaSuccess)
    {
      return stream_state::idle;
    }
    cudaGetLastError();
    return (status == cudaErrorNotReady) ? stream_state::busy : stream_state::unavailable;
  }

  void reclaim(int dev_id, size_t to_reclaim)
  {
    size_t reclaimed   = 0;
    auto& device_cache = cached_graphs[dev_id];

    // Reclaim the least-recently-used idle entries. cudaGraphExecDestroy must
    // not race an in-flight launch, so a busy entry remains cached even if
    // that temporarily leaves the cache above its configured size. An
    // `unavailable` entry (bound stream destroyed) is skipped too: its final
    // launch may still be draining, so destroying the executable is not
    // provably safe, and the entry stays as an unreclaimable zombie. This is
    // benign when cache-bound streams outlive the cache (the pool streams
    // handed out by async_resources_handle do); binding entries to streams
    // with independent lifetimes is what makes zombies possible at all.
    while (reclaimed < to_reclaim)
    {
      auto victim = device_cache.end();
      for (auto it = device_cache.begin(); it != device_cache.end(); ++it)
      {
        if (query_stream_state(it->second.stream) == stream_state::idle
            && (victim == device_cache.end() || it->second.last_use < victim->second.last_use))
        {
          victim = it;
        }
      }

      if (victim == device_cache.end())
      {
        break;
      }

      reclaimed += victim->second.footprint;
      total_cache_footprint[dev_id] -= victim->second.footprint;
      device_cache.erase(victim);
    }
  }

  // cached graphs index per device, then index per pair of edge/vertex count within each device
  ::std::vector<per_device_map_t> cached_graphs;

  // To keep track of the last recently used entries, we have an entry of
  size_t index = 0;

  // An estimated footprint (per device)
  ::std::vector<size_t> total_cache_footprint;

  size_t cache_size_limit;

  // A handle may be shared by multiple host threads. Serialize cache lookup,
  // update, insertion, and reclaim so one executable cannot be updated by two
  // queries concurrently.
  ::std::mutex mutex;
};
} // namespace cuda::experimental::stf
