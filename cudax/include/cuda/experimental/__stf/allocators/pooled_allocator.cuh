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
 * @brief Allocation strategy which suballocate entries from large pieces of memory
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
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/utility/pretty_print.cuh>

#include <optional>

namespace cuda::experimental::stf
{

namespace reserved
{

class block_data_pool_set;

/**
 * @brief A set of blocks (of the same size) of predetermined size located on a
 * specific data place. It is possible to have multiple of these per data
 * place.
 */
class block_data_pool
{
  /// One block that can be used for an allocation
  struct metadata
  {
    event_list prereqs;
    bool used = false;
  };

public:
  block_data_pool(
    backend_ctx_untyped& ctx,
    block_allocator_untyped root_allocator_,
    event_list& prereqs,
    data_place place,
    const size_t block_size,
    size_t nentries,
    double max_ratio)
      : place(mv(place))
      , block_size(block_size)
      , root_allocator(mv(root_allocator_))
  {
    if (this->place.is_host() || this->place.is_managed())
    {
      /* Pinned memory is not cheap, so we currently only allocate 4 blocks (arbitrarily) */
      nentries = 4;
    }
    else
    {
      const int dev = device_ordinal(this->place);
      assert(dev >= 0);
      cudaDeviceProp prop;
      cuda_safe_call(cudaGetDeviceProperties(&prop, dev));

      size_t max_mem = prop.totalGlobalMem;

      // We cap the memory at a certain fraction of total memory
      if (getenv("USER_ALLOC_POOLS_MEM_CAP"))
      {
        max_ratio = atof(getenv("USER_ALLOC_POOLS_MEM_CAP"));
      }

      if (nentries * block_size > (size_t) (max_ratio * max_mem))
      {
        nentries = size_t((max_ratio * max_mem) / block_size);
        fprintf(stderr,
                "Capping pool size at %f %%: nentries = %zu of %s...\n",
                max_ratio * 100.0,
                nentries,
                pretty_print_bytes(block_size).c_str());
      }
    }

    ::std::ptrdiff_t sz = nentries * block_size;
    base                = root_allocator.allocate(ctx, place, sz, prereqs);
    assert(sz > 0);
    assert(base);

    allocated_size = sz;

    entries.resize(nentries);
    for (auto& e : entries)
    {
      // Every entry depends on the allocation of the underlying buffer
      e.prereqs.merge(prereqs);
    }

    available_cnt = nentries;
  }

  /// Find one entry that is not used, sets the input/output event
  /// dependencies, and give a pointer to the corresponding chunk of memory
  void* get_entry(event_list& prereqs)
  {
    metadata* e = find_avail_pool_entry();

    if (!e)
    {
      assert(available_cnt == 0);
      return nullptr;
    }

    // We do not rely on a specific stream here so we do not join but
    // simply use existing deps to avoid useless hw dependencies
    prereqs.merge(e->prereqs);

    // This list will be used again when deallocating the entry, but we
    // have transferred existing dependencies to the allocate method
    e->prereqs.clear();

    assert(available_cnt > 0);
    available_cnt--;

    // Compute and return the pointer to data
    const ::std::ptrdiff_t offset = e - &entries[0];
    assert(offset >= 0 && offset < ::std::ptrdiff_t(entries.size()));
    return static_cast<char*>(base) + offset * block_size;
  }

  void release_entry(const event_list& prereqs, void* ptr)
  {
    // Find the index of the entry given the offset between ptr and base addresses
    const ptrdiff_t offset = static_cast<char*>(ptr) - static_cast<char*>(base);
    EXPECT(offset >= 0);
    const size_t index = size_t(offset) / block_size;
    auto& e            = entries[index];

    EXPECT(index < entries.size());
    EXPECT(e.used);
    EXPECT(e.prereqs.size() == 0);

    // To reuse this buffer, we need to wait for all users to finish using it
    e.prereqs.merge(prereqs);
    e.used = false;

    available_cnt++;

    // Set the last found to this, it will be next available
    //        last_found = index;
  }

  // This indicates whether a pointer corresponds to an entry in this pool
  bool is_inside(void* ptr) const
  {
    const ptrdiff_t offset = static_cast<char*>(ptr) - static_cast<char*>(base);
    // Add an assertion that offset % block_size == 0 ?
    return offset >= 0 && offset < static_cast<ptrdiff_t>(entries.size() * block_size);
  }

  // Free the memory allocated for this block of allocations
  void unpopulate(backend_ctx_untyped& ctx, event_list& prereqs)
  {
    root_allocator.deallocate(ctx, place, prereqs, base, allocated_size);
  }

private:
  // Find one entry that is not used
  metadata* find_avail_pool_entry()
  {
    for (auto i : each(0, entries.size()))
    {
      // Start from the item after the last found : the rationale is that
      // we want to avoid using a buffer that was recently discarded to
      // prevent implicit dependencies between presumably independent
      // tasks using the same memory buffers
      size_t j = 1 + last_found + i;
      if (j >= entries.size())
      {
        j -= entries.size();
      }
      assert(j < entries.size());
      if (!entries[j].used)
      {
        last_found      = j;
        entries[j].used = true;
        return &entries[j];
      }
    }
    return nullptr;
  }

  // Cache the last entry found
  size_t last_found = 0;

  // Number of unallocated blocks
  size_t available_cnt = 0;

  // Start of memory hunk
  void* base            = nullptr;
  size_t allocated_size = 0;

  // Memory belongs here
  data_place place;
  // Each block in the hunk has this size
  size_t block_size = 0;
  // One entry per block
  ::std::vector<metadata> entries;

  // The allocator used to create the base blocks
  block_allocator_untyped root_allocator;

  // To enable access to entries
  friend class block_data_pool_set;
};

} // end namespace reserved

/**
 * @brief Optional configurations to tune pooled allocators
 */
struct pooled_allocator_config
{
  // Maximum number of allocations per data place
  ::std::optional<size_t> max_entries_per_place;

  // Maximum amount of memory allocated per data place (as a ratio with the
  // total amount of memory)
  ::std::optional<double> max_ratio;

  // Maximum number of bytes allocated per data place
  ::std::optional<size_t> max_footprint_per_place;

  size_t get_max_entries_per_place() const
  {
    return max_entries_per_place.has_value() ? max_entries_per_place.value() : ::std::numeric_limits<size_t>::max();
  }

  double get_max_ratio() const
  {
    return max_ratio.has_value() ? max_ratio.value() : 0.9;
  }
};

namespace reserved
{

/// This class implements a set of blocks of allocations
class block_data_pool_set
{
public:
  block_data_pool_set() = default;
  // Disable copy constructor and assignment operator
  block_data_pool_set(const block_data_pool_set&)            = delete;
  block_data_pool_set& operator=(const block_data_pool_set&) = delete;

  void* get_pool_entry(backend_ctx_untyped& ctx,
                       block_allocator_untyped& root_allocator,
                       const data_place& memory_node,
                       size_t block_size,
                       event_list& prereqs)
  {
    // Get the pool or create it lazily
    block_data_pool* pool = get_pool(ctx, root_allocator, prereqs, memory_node, block_size);
    if (!pool)
    {
      return nullptr;
    }
    // Select an entry from it
    return pool->get_entry(prereqs);
  }

  void release_pool_entry(const data_place& memory_node, size_t block_size, const event_list& prereqs, void* ptr)
  {
    const size_t padded_sz = next_power_of_two(block_size);
    const auto key         = ::std::make_pair(to_index(memory_node), padded_sz);

    auto range = map.equal_range(key);

    /* Try to find an existing allocation that has at least an available slot */
    for (auto it = range.first; it != range.second; ++it)
    {
      if (it->second.is_inside(ptr))
      {
        it->second.release_entry(prereqs, ptr);
        return;
      }
    }

    // Should not be reached
    fprintf(stderr, "Error: pointer %p was released, but does not belong to a known pool.\n", ptr);
    abort();
  }

  event_list deinit_pools(backend_ctx_untyped& ctx)
  {
    event_list res;
    for (auto& entry : map)
    {
      event_list per_place_res;

      block_data_pool& p = entry.second;
      // fprintf(stderr, "UNPOPULATE pool %p for size %zu on node %zu\n", &entry.second, entry.first.first,
      //         entry.first.second);
      for (auto& m : p.entries)
      {
        per_place_res.merge(mv(m.prereqs));
      }

      p.unpopulate(ctx, per_place_res);

      res.merge(per_place_res);
    }

    return res;
  }

  void set_max_ratio(double r)
  {
    assert(r >= 0.0 && r <= 1.0);
    config.max_ratio = r;
  }

  void set_pool_size(size_t s)
  {
    assert(s > 0);
    config.max_entries_per_place = s;
  }

  void set_config(pooled_allocator_config _config)
  {
    config = mv(_config);
  }

private:
  /* Compute the power of 2 next to n (inclusive) */
  static size_t next_power_of_two(size_t n)
  {
    if (n <= 1)
    {
      return 1;
    }

    // Multiply by 2 until we reach a number greater or equal than n
    size_t powerOfTwo = 2;
    while (powerOfTwo < n)
    {
      powerOfTwo *= 2;
    }

    return powerOfTwo;
  }

  block_data_pool* get_pool(
    backend_ctx_untyped& ctx,
    block_allocator_untyped& root_allocator,
    event_list& prereqs,
    const data_place& memory_node,
    size_t sz)
  {
    const size_t padded_sz = next_power_of_two(sz);
    const auto key         = ::std::make_pair(to_index(memory_node), padded_sz);

    /* Try to find an existing allocation that has at least an available slot */
    auto range = map.equal_range(key);
    for (auto it : each(range.first, range.second))
    {
      if (it->second.available_cnt > 0)
      {
        return &it->second;
      }
    }

    /* There is no entry with an available slot so we need a new one */
    // Default number of entries per block of data
    return &map
              .emplace(key,
                       block_data_pool(
                         ctx,
                         root_allocator,
                         prereqs,
                         memory_node,
                         padded_sz,
                         config.get_max_entries_per_place(),
                         config.get_max_ratio()))
              ->second;
  }

  // For each memory node, a map of size_t to block_data_pool
  ::std::unordered_multimap<::std::pair<size_t, size_t>,
                            block_data_pool,
                            ::cuda::experimental::stf::hash<::std::pair<size_t, size_t>>>
    map;

private:
  pooled_allocator_config config = {
    .max_entries_per_place = 1024, .max_ratio = 0.2, .max_footprint_per_place = ::std::nullopt};
};

} // end namespace reserved

/**
 * @brief A pooled allocation strategy, where each memory node has a map, each
 * map associates a size_t with a pool of blocks of this size.
 *
 * This non static version uses different pools for the different instances of the allocator
 */
class pooled_allocator : public block_allocator_interface
{
public:
  // TODO pass two with std::optional or a structure pool_allocator_config with optionals ?
  pooled_allocator(block_allocator_untyped root_allocator_, double max_ratio = 0.2)
      : root_allocator(mv(root_allocator_))
  {
    pool_set.set_max_ratio(max_ratio);
  }
  pooled_allocator(double max_ratio = 0.2)
  {
    pool_set.set_max_ratio(max_ratio);
  }

  pooled_allocator(pooled_allocator_config config)
  {
    pool_set.set_config(mv(config));
  }

  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    auto* res = pool_set.get_pool_entry(
      ctx, root_allocator ? root_allocator : ctx.get_uncached_allocator(), memory_node, s, prereqs);
    if (res == nullptr)
    {
      s = -s;
    }
    return res;
  }

  void deallocate(
    backend_ctx_untyped&, const data_place& memory_node, event_list& prereqs, void* ptr, size_t block_size) override
  {
    pool_set.release_pool_entry(memory_node, block_size, prereqs, ptr);
  }

  event_list deinit(backend_ctx_untyped& ctx) override
  {
    return pool_set.deinit_pools(ctx);
  }

  ::std::string to_string() const override
  {
    return "pooled";
  }

private:
  reserved::block_data_pool_set pool_set;
  block_allocator_untyped root_allocator;
};

/**
 * @brief An allocator where all (preallocated) blocks have the same size
 *
 */
class fixed_size_allocator : public block_allocator_interface
{
public:
  fixed_size_allocator(size_t block_size)
      : block_size(block_size)
  {}

  fixed_size_allocator(size_t block_size, pooled_allocator_config config)
      : block_size(block_size)
  {
    pool_set.set_config(mv(config));
  }

  fixed_size_allocator() = delete;

  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    EXPECT(s <= block_size);
    auto* res = pool_set.get_pool_entry(ctx, ctx.get_uncached_allocator(), memory_node, block_size, prereqs);
    if (res == nullptr)
    {
      s = -s;
    }
    return res;
  }

  void
  deallocate(backend_ctx_untyped&, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) override
  {
    EXPECT(sz <= block_size);
    pool_set.release_pool_entry(memory_node, block_size, prereqs, ptr);
  }

  event_list deinit(backend_ctx_untyped& ctx) override
  {
    return pool_set.deinit_pools(ctx);
  }

  ::std::string to_string() const override
  {
    return "fixed size ";
  }

private:
  reserved::block_data_pool_set pool_set;

  // The (fixed) size of all blocks
  const size_t block_size;
};

} // end namespace cuda::experimental::stf
