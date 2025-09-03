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

#include <queue> // for ::std::priority_queue
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
  // Custom deleter specifically for cudaGraphExec_t
  auto cudaGraphExecDeleter = [](cudaGraphExec_t* pGraphExec) {
    cudaGraphExecDestroy(*pGraphExec);
  };

  ::std::shared_ptr<cudaGraphExec_t> res(new cudaGraphExec_t, cudaGraphExecDeleter);

  cuda_try(cudaGraphInstantiateWithFlags(res.get(), g, 0));

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

    int ndevices;
    cuda_safe_call(cudaGetDeviceCount(&ndevices));

    // One individual cache per device (TODO per execution place at some point
    // if we consider green contexts or multi-gpu graphs ?)
    cached_graphs.resize(ndevices);

    // Initialize the footprint per device too
    total_cache_footprint.resize(ndevices, 0);
  }

  // One entry of the cache
  struct entry
  {
    entry(executable_graph_cache* cache, ::std::shared_ptr<cudaGraphExec_t> exec_g_, size_t footprint)
        : cache(cache)
        , exec_g(mv(exec_g_))
        , footprint(footprint)
    {
      last_use = cache->index;
    }

    // Update the last_use field to mark that this entry was used recently
    void lru_refresh()
    {
      last_use = cache->index++;
    }

    executable_graph_cache* cache;
    ::std::shared_ptr<cudaGraphExec_t> exec_g;
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
  ::cuda::std::pair<::std::shared_ptr<cudaGraphExec_t>, bool>
  query(size_t nnodes, size_t nedges, ::std::shared_ptr<cudaGraph_t> g)
  {
    int dev_id = cuda_try<cudaGetDevice>();
    _CCCL_ASSERT(dev_id < int(cached_graphs.size()), "invalid device id value");

    auto range = cached_graphs[dev_id].equal_range({nnodes, nedges});
    for (auto it = range.first; it != range.second; ++it)
    {
      auto& e = it->second;
      if (reserved::try_updating_executable_graph(*e.exec_g, *g))
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

    auto exec_g = reserved::graph_instantiate(*g);

    // If we maintain a cache, store the executable graph
    if (cache_size_limit != 0)
    {
      cached_graphs[dev_id].insert({::std::make_pair(nnodes, nedges), entry(this, exec_g, footprint)});
      total_cache_footprint[dev_id] += footprint;
    }

    return ::cuda::std::make_pair(exec_g, false);
  }

private:
  void reclaim(int dev_id, size_t to_reclaim)
  {
    size_t reclaimed = 0;

    // Use a priority queue (min-heap) to track least recently used entries
    using key_type = ::std::pair<size_t, size_t>;

    auto& device_cache = cached_graphs[dev_id];

    auto cmp = [&device_cache](const key_type& key_a, const key_type& key_b) {
      auto iter_a = device_cache.find(key_a);
      auto iter_b = device_cache.find(key_b);

      // Directly compare last_use timestamps
      return iter_a->second.last_use > iter_b->second.last_use;
    };

    // Priority queue storing keys, ordered by least recently used
    ::std::priority_queue<key_type, ::std::vector<key_type>, decltype(cmp)> lru_queue(cmp);

    // Populate queue with keys from the cache
    for (const auto& kv : device_cache)
    {
      lru_queue.push(kv.first);
    }

    // Reclaim least recently used entries
    while (!lru_queue.empty() && reclaimed < to_reclaim)
    {
      key_type key = lru_queue.top();
      lru_queue.pop();

      // Find the entry before erasing
      auto it = device_cache.find(key);
      if (it != device_cache.end())
      {
        reclaimed += it->second.footprint;
        total_cache_footprint[dev_id] -= it->second.footprint;

        device_cache.erase(it);
      }
    }
  }

  // cached graphs index per device, then index per pair of edge/vertex count within each device
  ::std::vector<per_device_map_t> cached_graphs;

  // To keep track of the last recently used entries, we have an entry of
  size_t index = 0;

  // An estimated footprint (per device)
  ::std::vector<size_t> total_cache_footprint;

  size_t cache_size_limit;
};

} // namespace cuda::experimental::stf
