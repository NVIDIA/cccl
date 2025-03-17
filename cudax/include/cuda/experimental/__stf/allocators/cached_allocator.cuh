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
  using per_place_map_t = ::std::unordered_multimap<size_t, alloc_cache_entry>;

  /// Top-level cache map mapping data_place to per_place_map_t.
  ::std::unordered_map<data_place, per_place_map_t, hash<data_place>> free_cache;

  ::std::mutex allocator_mutex;
};

} // end namespace cuda::experimental::stf
