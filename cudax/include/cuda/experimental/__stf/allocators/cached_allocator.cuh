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
 * @brief Cached allocation strategy to reuse previously free'd buffers
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

#include <mutex>

namespace cuda::experimental::stf
{
/**
 * @brief Cached block allocator that implements block_allocator_interface.
 *
 * This allocator uses an internal cache to reuse previously allocated memory blocks.
 * The aim is to reduce the overhead of memory allocation.
 */
class cached_block_allocator : public block_allocator_interface
{
public:
  /**
   * @brief This constructor takes a root allocator which performs
   *        allocations when there is a cache miss
   */
  cached_block_allocator(block_allocator_untyped root_allocator_)
      : root_allocator(mv(root_allocator_))
  {}

  /**
   * @brief Allocates a memory block.
   *
   * Attempts to allocate a memory block from the internal cache if available.
   *
   * @param ctx The backend context (unused in this implementation).
   * @param memory_node Memory location where the allocation should happen.
   * @param s Pointer to the size of memory required.
   * @param prereqs List of events that this allocation depends on.
   * @return void* A pointer to the allocated memory block or nullptr if allocation fails.
   */
  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    EXPECT(s > 0);

    ::std::lock_guard<::std::mutex> g(allocator_mutex);
    if (auto it = free_cache.find(memory_node); it != free_cache.end())
    {
      per_place_map_t& m = it->second;
      if (auto it2 = m.find(s); it2 != m.end())
      {
        alloc_cache_entry& e = it2->second;
        prereqs.merge(mv(e.prereq));
        void* result = e.ptr;
        m.erase(it2);
        return result;
      }
    }

    // That is a miss, we need to allocate data using the root allocator
    return root_allocator.allocate(ctx, memory_node, s, prereqs);
  }

  /**
   * @brief Deallocates a memory block.
   *
   * Puts the deallocated block back into the internal cache for future reuse.
   *
   * @param ctx The backend context (unused in this implementation).
   * @param memory_node Memory location where the deallocation should happen.
   * @param prereqs List of events that this deallocation depends on.
   * @param ptr Pointer to the memory block to be deallocated.
   * @param sz Size of the memory block.
   */
  void
  deallocate(backend_ctx_untyped&, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) override
  {
    ::std::lock_guard<::std::mutex> g(allocator_mutex);
    // We do not call the deallocate method of the root allocator, we discard buffers instead
    free_cache[memory_node].emplace(sz, alloc_cache_entry{ptr, prereqs});
  }

  /**
   * @brief De-initializes the allocator.
   *
   */
  event_list deinit(backend_ctx_untyped& ctx) override
  {
    ::std::unordered_map<data_place, per_place_map_t, hash<data_place>> free_cache_janitor;
    {
      ::std::lock_guard<::std::mutex> g(allocator_mutex);
      free_cache.swap(free_cache_janitor);
    }

    event_list result;
    for (auto& entry : free_cache_janitor)
    {
      auto& where        = entry.first;
      per_place_map_t& m = entry.second;
      for (auto& entry2 : m)
      {
        size_t sz              = entry2.first;
        alloc_cache_entry& ace = entry2.second;

        root_allocator.deallocate(ctx, where, ace.prereq, ace.ptr, sz);

        // Move all events
        result.merge(mv(ace.prereq));
      }
    }
    return result;
  }

  /**
   * @brief Returns a string representation of the allocator.
   *
   * @return std::string The string "cached_block_allocator".
   */
  ::std::string to_string() const override
  {
    return "cached_block_allocator";
  }

protected:
  /**
   * @brief Underlying root allocator for base buffers
   */
  block_allocator_untyped root_allocator;

  /**
   * @brief Struct representing an entry in the cache.
   *
   * Holds the address of the allocated memory and a list of prerequisites for its reuse.
   */
  struct alloc_cache_entry
  {
    void* ptr; ///< Pointer to the allocated memory block.
    event_list prereq; ///< Prerequisites for reusing this block.
  };

  /// Maps sizes to cache entries for a given data_place.
  using per_place_map_t = ::std::unordered_multimap<size_t, alloc_cache_entry>;

  /// Top-level cache map mapping data_place to per_place_map_t.
  ::std::unordered_map<data_place, per_place_map_t, hash<data_place>> free_cache;

  ::std::mutex allocator_mutex;
};

/**
 * @brief A block allocator with FIFO caching strategy.
 *
 * This allocator wraps a root allocator and implements a caching mechanism
 * to efficiently reuse previously allocated memory blocks.
 *
 * When a memory block is requested, it first attempts to find a suitable block
 * in the internal cache (`free_cache`). If no suitable block is found, a larger
 * buffer is allocated from the root allocator, split into smaller blocks, and
 * stored in the cache for future reuse.
 *
 * The caching policy is FIFO (First-In-First-Out): freed blocks are appended
 * to a queue and reused in the order they were deallocated.
 *
 * At deinitialization (`deinit()`), all cached blocks and their associated large
 * allocations are properly released by delegating to the root allocator.
 *
 * @note The allocator is thread-safe and uses a mutex to protect access
 * to its internal cache.
 *
 * @note Deallocation does not immediately release memory back to the root allocator.
 * Instead, deallocated blocks are stored in the cache for future reuse.
 *
 * @see block_allocator_interface
 */
class cached_block_allocator_fifo : public block_allocator_interface
{
public:
  /**
   * @brief This constructor takes a root allocator which performs
   *        allocations when there is a cache miss
   */
  cached_block_allocator_fifo(block_allocator_untyped root_allocator_)
      : root_allocator(mv(root_allocator_))
  {}

  /**
   * @brief Allocates a memory block.
   *
   * Attempts to allocate a memory block from the internal cache if available.
   *
   * @param ctx The backend context (unused in this implementation).
   * @param memory_node Memory location where the allocation should happen.
   * @param s Pointer to the size of memory required.
   * @param prereqs List of events that this allocation depends on.
   * @return void* A pointer to the allocated memory block or nullptr if allocation fails.
   */
  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    EXPECT(s > 0);

    ::std::lock_guard<::std::mutex> g(allocator_mutex);
    if (auto it = free_cache.find(memory_node); it != free_cache.end())
    {
      per_place_map_t& size_map = it->second;
      if (auto it2 = size_map.find(s); it2 != size_map.end() && !it2->second.empty())
      {
        // Retrieve the first (oldest) entry.
        alloc_cache_entry e = std::move(it2->second.front());
        it2->second.pop();

        // Optionally, remove the key if the queue is now empty.
        if (it2->second.empty())
        {
          size_map.erase(it2);
        }

        prereqs.merge(std::move(e.prereq));
        return e.ptr;
      }
    }

    // That is a miss, we need to allocate data using the root allocator

    /* Create one large block of memory */
    static const size_t cnt = [] {
      const char* fifo_env = ::std::getenv("CUDASTF_CACHED_FIFO");
      return (fifo_env ? atol(fifo_env) : 50);
    }();

    ::std::ptrdiff_t large_sz = cnt * s;
    auto* base                = root_allocator.allocate(ctx, memory_node, large_sz, prereqs);
    _CCCL_ASSERT(large_sz >= 0, "failed to allocate large buffer");

    large_allocations[memory_node].push_back(::std::make_pair(base, large_sz));

    /* Populate the cache with sub-entries from this large block, except the
     * first entry which is used immediately */
    auto& per_node = free_cache[memory_node][s];
    for (size_t k = 1; k < cnt; k++)
    {
      per_node.push(alloc_cache_entry{(char*) base + s * k, prereqs});
    }

    return base;
  }

  /**
   * @brief Deallocates a memory block.
   *
   * Puts the deallocated block back into the internal cache for future reuse.
   *
   * @param ctx The backend context (unused in this implementation).
   * @param memory_node Memory location where the deallocation should happen.
   * @param prereqs List of events that this deallocation depends on.
   * @param ptr Pointer to the memory block to be deallocated.
   * @param sz Size of the memory block.
   */
  void
  deallocate(backend_ctx_untyped&, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) override
  {
    ::std::lock_guard<::std::mutex> g(allocator_mutex);
    // We do not call the deallocate method of the root allocator, we discard buffers instead.
    // Note that we do not need to keep track of the "large buffer" from which this originated
    free_cache[memory_node][sz].push(alloc_cache_entry{ptr, prereqs});
  }

  /**
   * @brief De-initializes the allocator.
   *
   */
  event_list deinit(backend_ctx_untyped& ctx) override
  {
    decltype(free_cache) free_cache_janitor;
    {
      ::std::lock_guard<::std::mutex> g(allocator_mutex);
      free_cache.swap(free_cache_janitor);
    }

    event_list result;
    for (auto& [where, size_map] : free_cache_janitor)
    {
      event_list per_place_result;

      // For each size key, iterate over the associated queue
      for (auto& [sz, queue] : size_map)
      {
        // We first go through all suballocations to "collect" prereqs
        while (!queue.empty())
        {
          alloc_cache_entry ace = std::move(queue.front());
          queue.pop();

          per_place_result.merge(mv(ace.prereq));
        }
      }

      // Then we actually deallocate memory: for each place, there is a vector of large allocations
      for (auto& alloc : large_allocations[where])
      {
        void* base      = alloc.first;
        size_t large_sz = alloc.second;
        root_allocator.deallocate(ctx, where, per_place_result, base, large_sz);
      }

      result.merge(mv(per_place_result));
    }
    return result;
  }

  /**
   * @brief Returns a string representation of the allocator.
   *
   * @return std::string The string "cached_block_allocator".
   */
  ::std::string to_string() const override
  {
    return "cached_block_allocator_fifo";
  }

  /**
   * @brief Prints additional information about the allocator.
   *
   * This function currently prints no additional information.
   */
  void print_info() const override
  {
    const auto s = to_string();
    fprintf(stderr, "No additional info for allocator of kind \"%.*s\".\n", static_cast<int>(s.size()), s.data());
  }

protected:
  /**
   * @brief Underlying root allocator for base buffers
   */
  block_allocator_untyped root_allocator;

  /**
   * @brief Struct representing an entry in the cache.
   *
   * Holds the address of the allocated memory and a list of prerequisites for its reuse.
   */
  struct alloc_cache_entry
  {
    void* ptr; ///< Pointer to the allocated memory block.
    event_list prereq; ///< Prerequisites for reusing this block.
  };

  /// Maps sizes to cache entries for a given data_place.
  using per_place_map_t = ::std::unordered_map<size_t, ::std::queue<alloc_cache_entry>>;

  /// We track actually allocated blocks (per data place), and then a map of
  // suballocations. The deallocate method only return ptr/sz so we loose track
  // of the connection between large blocks from the root allocator, and
  // small allocations, but we simply clear these blocks at the end.
  ::std::unordered_map<data_place, per_place_map_t, hash<data_place>> free_cache;
  ::std::unordered_map<data_place, ::std::vector<::std::pair<void*, size_t>>, hash<data_place>> large_allocations;

  ::std::mutex allocator_mutex;
};
} // end namespace cuda::experimental::stf
