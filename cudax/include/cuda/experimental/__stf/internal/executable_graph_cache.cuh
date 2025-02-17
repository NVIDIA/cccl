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
#if _CCCL_CUDACC_BELOW(12)
  cudaGraphNode_t errorNode;
  cudaGraphExecUpdateResult updateResult;
  cudaGraphExecUpdate(exec_graph, graph, &errorNode, &updateResult);
#else
  cudaGraphExecUpdateResultInfo resultInfo;
  cudaGraphExecUpdate(exec_graph, graph, &resultInfo);
#endif

  // Be sure to "erase" the last error
  cudaError_t res = cudaGetLastError();

#ifdef CUDASTF_DEBUG
  reserved::counter<reserved::graph_tag::update>.increment();
  if (res == cudaSuccess)
  {
    reserved::counter<reserved::graph_tag::update::success>.increment();
  }
  else
  {
    reserved::counter<reserved::graph_tag::update::failure>.increment();
  }
#endif

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

#ifdef CUDASTF_DEBUG
  reserved::counter<reserved::graph_tag::instantiate>.increment();
#endif

  return res;
}

} // end namespace reserved

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
  ::std::shared_ptr<cudaGraphExec_t> query(size_t nnodes, size_t nedges, ::std::shared_ptr<cudaGraph_t> g)
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
        return e.exec_g;
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

    return exec_g;
  }

private:
  void reclaim(int dev_id, size_t to_reclaim)
  {
    size_t reclaimed = 0;

    // Use a priority queue (min-heap) to track least recently used entries
    using entry_iter = per_device_map_t::iterator;
    auto cmp         = [](const entry_iter& a, const entry_iter& b) {
      return a->second.last_use > b->second.last_use;
    };
    ::std::priority_queue<entry_iter, ::std::vector<entry_iter>, decltype(cmp)> lru_queue(cmp);

    // Populate the priority queue with all cache entries
    for (auto it = cached_graphs[dev_id].begin(); it != cached_graphs[dev_id].end(); ++it)
    {
      lru_queue.push(it);
    }

    // Remove LRU entries until we've reclaimed enough space
    while (!lru_queue.empty() && reclaimed < to_reclaim)
    {
      auto lru_it = lru_queue.top();
      lru_queue.pop();

      reclaimed += lru_it->second.footprint;
      total_cache_footprint[dev_id] -= lru_it->second.footprint;
      cached_graphs[dev_id].erase(lru_it);
    }

#if 0
    fprintf(stderr,
            "Reclaimed %s in cache graph (asked %s remaining %s)\n",
            pretty_print_bytes(reclaimed).c_str(),
            pretty_print_bytes(to_reclaim).c_str(),
            pretty_print_bytes(total_cache_footprint[dev_id]).c_str());
#endif
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
