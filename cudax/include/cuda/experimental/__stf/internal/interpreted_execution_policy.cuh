//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/cooperative_group_system.cuh>
#include <cuda/experimental/__stf/internal/execution_policy.cuh>

#include <vector>

namespace cuda::experimental::stf
{
class exec_place;

template <auto... spec>
class thread_hierarchy;

/**
 * This corresponds to an execution_policy_spec (e.g. par(con(32))) which we
 * map on an execution place.
 *
 * The goal of this intermediate class is to take the high-level spec
 * description and compute how the different levels are mapped on the machine.
 * In particular, how levels are mapped to the CUDA hierarchy (threads, blocks,
 * devices).
 */
template <auto... spec>
class interpreted_execution_policy
{
  static constexpr size_t depth = sizeof...(spec) / 2;
  static_assert(sizeof...(spec) % 2 == 0, "Number of template arguments must be even.");

public:
  using thread_hierarchy_t = thread_hierarchy<spec...>;

  /**
   * Each level of the interpreted policy is a vector which describes how the
   * level is spread across the machine. For example, a level could be (128
   * threads), or it could be (4 blocks) x (32 threads). In the latter
   * example, the level is described as a vector of 2 subentries.
   */
  class level
  {
  public:
    level(::std::pair<hw_scope, size_t> desc, size_t local_mem = 0)
        : local_mem(local_mem)
    {
      level_desc.push_back(mv(desc));
    }
    level(::std::initializer_list<::std::pair<hw_scope, size_t>> desc, size_t local_mem = 0)
        : level_desc(desc.begin(), desc.end())
        , local_mem(local_mem)
    {}

    // Get the total width of the level, which is the product of all sublevel sizes
    size_t width() const
    {
      size_t res = 1;
      for (auto& e : level_desc)
      {
        res *= e.second;
      }
      return res;
    }

    void set_mem(size_t size)
    {
      local_mem = size;
    }
    size_t get_mem() const
    {
      return local_mem;
    }

    void set_sync(bool sync)
    {
      may_sync = sync;
    }
    bool get_sync() const
    {
      return may_sync;
    }

    const auto& get_desc() const
    {
      return level_desc;
    }

  private:
    ::std::vector<::std::pair<hw_scope, size_t>> level_desc;
    size_t local_mem;
    bool may_sync = false;
  };

public:
  interpreted_execution_policy()  = default;
  ~interpreted_execution_policy() = default;

  /* A substem that contains sync() related functionality for the thread hierarchy */
  reserved::cooperative_group_system cg_system;

  template <typename Fun>
  interpreted_execution_policy(const thread_hierarchy_spec<spec...>& p, const exec_place& where, const Fun& f);

  void add_level(level l)
  {
    levels.push_back(mv(l));
  }

  void set_level_mem(int level, size_t size)
  {
    assert(level < int(depth));
    levels[level].set_mem(size);
  }

  size_t get_level_mem(int level) const
  {
    assert(level < depth);
    return levels[level].get_mem();
  }

  void set_level_sync(int level, bool sync)
  {
    assert(level < int(depth));
    levels[level].set_sync(sync);
  }

  bool get_level_sync(size_t level) const
  {
    assert(level < int(depth));
    return levels[level].get_sync();
  }

  // Returns the width of a level
  size_t width(int l) const
  {
    EXPECT(static_cast<size_t>(l) < depth);
    return levels[l].width();
  }

  const auto& get_levels() const
  {
    return levels;
  }

  /* Compute the kernel configuration given all levels */
  ::std::array<size_t, 3> get_config() const
  {
    ::std::array<size_t, 3> config = {1, 1, 1};

    // Scan all subsets of all levels. Consider three levels, level0 = (4
    // GPUs) level1 = (32 blocks x 4 threads) level 2= (32 threads). Then
    // we would have an overall configuration of 4 GPUs, 32 blocks and
    // 4*32=128 threads.
    for (auto& l : levels)
    {
      for (auto& e : l.get_desc())
      {
        switch (e.first)
        {
          case hw_scope::device:
            config[0] *= e.second;
            break;

          case hw_scope::block:
            config[1] *= e.second;
            break;

          case hw_scope::thread:
            config[2] *= e.second;
            break;

          default:
            assert(!"Corrupt hw_scope value.");
            abort();
        }
      }
    }

    return config;
  }

  /* Helper function to return the hw scope of the last level in the interpreted thread hierarchy. Probably a more
   * elegant and safer way to do the below exists. */
  hw_scope last_level_scope() const
  {
    return (levels[0].get_desc())[0].first;
  }

  /* Compute overall memory requirements */
  ::std::array<size_t, 3> get_mem_config() const
  {
    /* We take the kernel configuration : config[2] is the number of
     * threads per block, config[1] the number of blocks, config[0] the
     * number of devices on which this kernel is launched. The total number
     * of threads per scope corresponds to the product of these values. On
     * each device, we for example get config[1] * config[2] threads, which
     * is the product of the number of blocks,and the number of threads per
     * block. */
    ::std::array<size_t, 3> config = get_config();
    size_t system_scope_size       = config[0] * config[1] * config[2];
    size_t device_scope_size       = config[1] * config[2];
    size_t block_scope_size        = config[2];

    ::std::array<size_t, 3> mem_config = {0, 0, 0};

    size_t width_product = 1;

    // Scan all levels in reversed order : we look for the number of
    // threads contained in a level : the memory assigned to a level
    // corresponds to the hardware level which "contains" this level. For
    // example, if a level has the same number of threads as the device,
    // this means we are going to allocate device memory for this
    // scratchpad.
    for (size_t i = levels.size(); i-- > 0;)
    {
      auto& l  = levels[i];
      size_t w = l.width();
      width_product *= w;

      size_t m = l.get_mem();
      if (m == 0)
      {
        continue;
      }

      // We only support levels that match boundaries of hw scopes for now
      EXPECT((width_product == system_scope_size || width_product == device_scope_size
              || width_product == block_scope_size));

      /* Find which hardware level matches the size of that spec level */
      if (width_product == block_scope_size)
      {
        mem_config[2] += m;
      }
      else if (width_product == device_scope_size)
      {
        mem_config[1] += m;
      }
      else
      {
        EXPECT(width_product == system_scope_size);
        mem_config[0] += m;
      }
    }

    return mem_config;
  }

  bool need_cooperative_kernel_launch() const
  {
    ::std::array<size_t, 3> config = get_config();
    size_t block_scope_size        = config[2];
    size_t width_product           = 1;

    // Scan all levels in reversed order
    for (size_t i = levels.size(); i-- > 0;)
    {
      auto& l  = levels[i];
      size_t w = l.width();
      width_product *= w;

      if (width_product > block_scope_size && get_level_sync(i))
      {
        return true;
      }
    }

    return false;
  }

  void set_system_mem(void* addr)
  {
    system_mem = addr;
  }
  void* get_system_mem() const
  {
    return system_mem;
  }

private:
  ::std::vector<level> levels;

  // Stored for convenience here
  void* system_mem = nullptr;
};

#ifdef UNITTESTED_FILE
UNITTEST("interpreted policy")
{
  interpreted_execution_policy<false, size_t(0), true, size_t(0)> p;

  // par(2)
  p.add_level({::std::make_pair(hw_scope::device, 2UL)});
  p.set_level_mem(0, 0);
  p.set_level_sync(0, false);

  // con(16*32)
  p.add_level({::std::make_pair(hw_scope::block, 16UL), ::std::make_pair(hw_scope::thread, 32UL)});
  p.set_level_mem(1, 0);
  p.set_level_sync(1, true);
};
#endif // UNITTESTED_FILE
} // end namespace cuda::experimental::stf
