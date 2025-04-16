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
 * @brief Buddy allocator implementation
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
#include <cuda/experimental/__stf/internal/async_prereq.cuh>
#include <cuda/experimental/__stf/internal/backend_ctx.cuh>
#include <cuda/experimental/__stf/utility/pretty_print.cuh>

namespace cuda::experimental::stf
{

namespace reserved
{

/**
 * @brief Buddy allocator for one data place
 *
 * It does not manipulate memory at all, but returns offsets within some memory space of size "size"
 *
 * We (currently) assume that the size is a power of 2
 */
class buddy_allocator_metadata
{
private:
  /**
   * @brief Describes an available piece of memory
   */
  struct avail_block
  {
    // Copy prereqs
    avail_block(size_t index_, event_list prereqs_)
        : index(index_)
        , prereqs(mv(prereqs_))
    {}

    size_t index = 0; // location of the block
    event_list prereqs; // dependencies to use that block
  };

public:
  buddy_allocator_metadata(size_t size, event_list init_prereqs)
      : free_lists_(int_log2(next_power_of_two(size)) + 1)
  {
    _CCCL_ASSERT(size && (size & (size - 1)) == 0,
                 "Allocation requests for this allocator must pass a size that is a power of two.");
    // Initially, the whole memory is free, but depends on init_prereqs
    free_lists_.back().emplace_back(0, mv(init_prereqs));
  }

  ::std::ptrdiff_t allocate(size_t size, event_list& prereqs)
  {
    size               = next_power_of_two(size);
    const size_t level = int_log2(size);
    if (level >= free_lists_.size())
    {
      fprintf(stderr, "Level %zu > max level %zu\n", level, free_lists_.size() - 1);
      return -1;
    }

    ::std::ptrdiff_t alloc_index = find_free_block(level, prereqs);
    if (alloc_index == -1)
    {
      fprintf(stderr, "No free block available for size %zu\n", size);
      return -1;
    }

    return alloc_index;
  }

  void deallocate(::std::ptrdiff_t index, size_t size, event_list& prereqs)
  {
    size         = next_power_of_two(size);
    size_t level = int_log2(size);

    // Deallocated blocks will depend on these, and we will merge the
    // previous dependencies when merging buddies
    event_list block_prereqs(prereqs);
    const size_t max_level = free_lists_.size() - 1;
    while (level < max_level)
    {
      const size_t buddy_index = get_buddy_index(index, level);
      auto& buddy_list         = free_lists_[level];
      auto it = ::std::find_if(buddy_list.begin(), buddy_list.end(), [buddy_index](const avail_block& block) {
        return block.index == buddy_index;
      });

      if (it == buddy_list.end())
      {
        // No buddy available to merge, stop here
        break;
      }
      // Merge with buddy
      block_prereqs.merge(it->prereqs);

      buddy_list.erase(it);
      index = ::std::min(index, ::std::ptrdiff_t(buddy_index));
      level++;
    }

    free_lists_[level].emplace_back(index, prereqs);
  }

  void deinit(event_list& prereqs)
  {
    for (auto& level : free_lists_)
    {
      for (auto& block : level)
      {
        prereqs.merge(block.prereqs);
        block.prereqs.clear();
      }
    }
  }

  void debug_print() const
  {
    size_t power = 1;
    for (size_t i = 0; i < free_lists_.size(); ++i, power *= 2)
    {
      if (!free_lists_[i].empty())
      {
        fprintf(stderr, "Level %zu : %s bytes : ", i, pretty_print_bytes(power).c_str());
        for (const auto& b : free_lists_[i])
        {
          fprintf(stderr, "[%zu, %zu[ ", b.index, b.index + power);
        }
        fprintf(stderr, "\n");
      }
    }
  }

private:
  static size_t next_power_of_two(size_t size)
  {
    static_assert(sizeof(size_t) <= 8, "You must be from the future. Review and adjust this code.");
    if (size == 0)
    {
      return 1;
    }
    --size;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    if constexpr (sizeof(size_t) == 8)
    {
      size |= size >> 32;
    }
    return size + 1;
  }

  static size_t int_log2(size_t n)
  {
    assert(n > 0);

    size_t log = 0;
    while (n >>= 1)
    { // Right shift until n becomes 0
      log++;
    }
    return log;
  }

  ::std::ptrdiff_t find_free_block(size_t level, event_list& prereqs)
  {
    for (size_t current_level : each(level, free_lists_.size()))
    {
      if (free_lists_[current_level].empty())
      {
        continue;
      }
      auto& b              = free_lists_[current_level].back();
      size_t block_index   = b.index;
      event_list b_prereqs = mv(b.prereqs);
      free_lists_[current_level].pop_back();

      // Dependencies to reuse that block
      prereqs.merge(b_prereqs);

      // If we are not at the requested level, split blocks
      while (current_level > level)
      {
        current_level--;
        size_t buddy_index = block_index + (1ull << current_level);
        // split blocks depend on the previous dependencies of the whole unsplit block
        free_lists_[current_level].emplace_back(buddy_index, b_prereqs);
      }
      return block_index;
    }

    return -1; // No block available
  }

  size_t get_buddy_index(size_t index, size_t level)
  {
    assert(level <= 63);
    return index ^ (1ull << level); // XOR to find the buddy block
  }

  ::std::vector<::std::vector<avail_block>> free_lists_;
};

} // end namespace reserved

/**
 * @brief Buddy allocator policy which relies on a root allocator to create
 * large buffers which are then suballocated with a buddy allocation algorithm
 */
class buddy_allocator : public block_allocator_interface
{
public:
  buddy_allocator() = default;
  buddy_allocator(block_allocator_untyped root_allocator_)
      : root_allocator(mv(root_allocator_))
  {}

private:
  // Per data place buffer and its corresponding metadata
  struct per_place
  {
    per_place(void* base_, size_t size, event_list prereqs)
        : base(base_)
        , buffer_size(size)
        , metadata(buffer_size, mv(prereqs))
    {}

    per_place& operator=(const per_place&) = delete;
    per_place& operator=(per_place&&)      = default;
    per_place(const per_place&)            = delete;
    per_place(per_place&&)                 = default;

    void* base         = nullptr;
    size_t buffer_size = 0;
    reserved::buddy_allocator_metadata metadata;
  };

public:
  void*
  allocate(backend_ctx_untyped& ctx, const data_place& memory_node, ::std::ptrdiff_t& s, event_list& prereqs) override
  {
    auto it = map.find(memory_node);
    if (it == map.end())
    {
      // There is currently no buffer associated to this place, create one lazily
      // 1. create memory on that place
      ::std::ptrdiff_t sz = 128 * 1024 * 1024; // TODO
      auto& a             = root_allocator ? root_allocator : ctx.get_uncached_allocator();
      void* base          = a.allocate(ctx, memory_node, sz, prereqs);

      // 2. creates meta data for that buffer, and 3. associate it to the data place
      it = map.emplace(memory_node, per_place(base, sz, prereqs)).first;
    }

    // There should be exactly one entry in the map
    assert(map.count(memory_node) == 1);
    auto& m = it->second;

    ::std::ptrdiff_t offset = m.metadata.allocate(s, prereqs);
    assert(offset != -1);
    return static_cast<char*>(m.base) + offset;
  }

  void
  deallocate(backend_ctx_untyped&, const data_place& memory_node, event_list& prereqs, void* ptr, size_t sz) override
  {
    // There should be exactly one entry in the map
    assert(map.count(memory_node) == 1);
    auto& m = map.find(memory_node)->second;

    size_t offset = static_cast<char*>(ptr) - static_cast<char*>(m.base);

    m.metadata.deallocate(offset, sz, prereqs);
  }

  event_list deinit(backend_ctx_untyped& ctx) override
  {
    event_list result;
    // For every place in the map
    for (auto& [memory_node, pp] : map)
    {
      event_list local_prereqs;

      // Deinitialize the metadata of the buddy allocator for this place
      pp.metadata.deinit(local_prereqs);

      // Deallocate the underlying buffer for this buddy allocator
      auto& a = root_allocator ? root_allocator : ctx.get_uncached_allocator();
      a.deallocate(ctx, memory_node, local_prereqs, pp.base, pp.buffer_size);

      result.merge(local_prereqs);
    }
    return result;
  }

  ::std::string to_string() const override
  {
    return "buddy allocator";
  }

private:
  ::std::unordered_map<data_place, per_place, hash<data_place>> map;

  block_allocator_untyped root_allocator;
};

#ifdef UNITTESTED_FILE

UNITTEST("buddy_allocator is movable")
{
  static_assert(std::is_move_constructible<buddy_allocator>::value, "buddy_allocator must be move constructible");
  static_assert(std::is_move_assignable<buddy_allocator>::value, "buddy_allocator must be move assignable");
};

UNITTEST("buddy allocator meta data")
{
  event_list prereqs; // starts empty

  reserved::buddy_allocator_metadata allocator(1024, prereqs);

  // ::std::cout << "Initial state:" << ::std::endl;
  // allocator.debug_print();

  event_list dummy;

  ::std::ptrdiff_t ptr1 = allocator.allocate(200, dummy); // Allocate 200 bytes
  // ::std::cout << "\nAfter allocating 200 bytes:" << ::std::endl;
  // allocator.debug_print();

  ::std::ptrdiff_t ptr2 = allocator.allocate(300, dummy); // Allocate 300 bytes
  // ::std::cout << "\nAfter allocating 300 bytes:" << ::std::endl;
  // allocator.debug_print();

  allocator.deallocate(ptr1, 200, dummy); // Free the 200 bytes
  // ::std::cout << "\nAfter freeing 200 bytes:" << ::std::endl;
  // allocator.debug_print();

  allocator.deallocate(ptr2, 300, dummy); // Free the 300 bytes
  // ::std::cout << "\nAfter freeing 300 bytes:" << ::std::endl;
  // allocator.debug_print();
};

#endif // UNITTESTED_FILE

} // end namespace cuda::experimental::stf
