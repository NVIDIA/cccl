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
 *
 * @brief Deferred definition of the interpreted_execution_policy constructor.
 *
 * The constructor maps a thread_hierarchy_spec onto an exec_place, which
 * requires both exec_place (from places.cuh) and compute_kernel_limits
 * (from occupancy.cuh) to be fully defined. This separate file breaks the
 * circular dependency between interpreted_execution_policy.cuh and places.cuh.
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

#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/internal/interpreted_execution_policy.cuh>
#include <cuda/experimental/__stf/utility/occupancy.cuh>

namespace cuda::experimental::stf
{
template <auto... spec>
template <typename Fun>
interpreted_execution_policy<spec...>::interpreted_execution_policy(
  const thread_hierarchy_spec<spec...>& p, const exec_place& where, const Fun& f)
{
  constexpr size_t pdepth = sizeof...(spec) / 2;

  if (where == exec_place::host())
  {
    // XXX this may not match the type of the spec if we are not using the default spec ...
    for (size_t d = 0; d < pdepth; d++)
    {
      this->add_level({::std::make_pair(hw_scope::thread, 1)});
    }
    return;
  }

  size_t ndevs = where.size();

  if constexpr (pdepth == 1)
  {
    size_t l0_size = p.get_width(0);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;

    size_t shared_mem_bytes = 0;

    auto kernel_limits = reserved::compute_kernel_limits(f, shared_mem_bytes, l0_sync);

    int grid_size = 0;
    int block_size;

    if (l0_size == 0)
    {
      grid_size = kernel_limits.min_grid_size;
      // Maximum occupancy without exceeding limits
      block_size = ::std::min(kernel_limits.max_block_size, kernel_limits.block_size_limit);
      l0_size    = ndevs * grid_size * block_size;
    }
    else
    {
      // Find grid_size and block_size such that grid_size*block_size = l0_size and block_size <= max_block_size
      for (block_size = kernel_limits.max_block_size; block_size >= 1; block_size--)
      {
        if (l0_size % block_size == 0)
        {
          grid_size = l0_size / block_size;
          break;
        }
      }
    }

    // Make sure we have computed the width if that was implicit
    _CCCL_ASSERT(l0_size > 0, "invalid level 0 size");

    _CCCL_ASSERT(grid_size > 0, "invalid grid size");
    _CCCL_ASSERT(block_size <= kernel_limits.max_block_size, "invalid block size");

    _CCCL_ASSERT(l0_size % ndevs == 0, "invalid level 0 size");
    _CCCL_ASSERT(l0_size % (ndevs * block_size) == 0, "invalid level 0 size");

    _CCCL_ASSERT(ndevs * grid_size * block_size == l0_size, "invalid level 0 size");

    this->add_level({::std::make_pair(hw_scope::device, ndevs),
                     ::std::make_pair(hw_scope::block, grid_size),
                     ::std::make_pair(hw_scope::thread, block_size)});
    this->set_level_mem(0, size_t(p.get_mem(0)));
    this->set_level_sync(0, l0_sync);
  }
  else if constexpr (pdepth == 2)
  {
    size_t l0_size = p.get_width(0);
    size_t l1_size = p.get_width(1);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;
    bool l1_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<1>;

    /* level 1 will be mapped on threads, level 0 on blocks and above */
    size_t shared_mem_bytes = size_t(p.get_mem(1));
    auto kernel_limits      = reserved::compute_kernel_limits(f, shared_mem_bytes, l0_sync);

    // For implicit widths, use sizes suggested by CUDA occupancy calculator
    if (l1_size == 0)
    {
      // Maximum occupancy without exceeding limits
      l1_size = ::std::min(kernel_limits.max_block_size, kernel_limits.block_size_limit);
    }
    else
    {
      if (int(l1_size) > kernel_limits.block_size_limit)
      {
        fprintf(stderr,
                "Unsatisfiable spec: Maximum block size %d threads, requested %zu (level 1)\n",
                kernel_limits.block_size_limit,
                l1_size);
        abort();
      }
    }

    if (l0_size == 0)
    {
      l0_size = kernel_limits.min_grid_size * ndevs;
    }

    // Enforce the resource limits in the number of threads per block
    _CCCL_ASSERT(int(l1_size) <= kernel_limits.block_size_limit, "invalid level 1 size");
    _CCCL_ASSERT(l0_size % ndevs == 0, "invalid level 0 size");

    /* Merge blocks and devices */
    this->add_level({::std::make_pair(hw_scope::device, ndevs), ::std::make_pair(hw_scope::block, l0_size / ndevs)});
    this->set_level_mem(0, size_t(p.get_mem(0)));
    this->set_level_sync(0, l0_sync);

    this->add_level({::std::make_pair(hw_scope::thread, l1_size)});
    this->set_level_mem(1, size_t(p.get_mem(1)));
    this->set_level_sync(1, l1_sync);
  }
  else if constexpr (pdepth == 3)
  {
    size_t l0_size = p.get_width(0);
    size_t l1_size = p.get_width(1);
    size_t l2_size = p.get_width(2);
    bool l0_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<0>;
    bool l1_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<1>;
    bool l2_sync   = thread_hierarchy_spec<spec...>::template is_synchronizable<2>;

    /* level 2 will be mapped on threads, level 1 on blocks, level 0 on devices */
    size_t shared_mem_bytes = size_t(p.get_mem(2));
    auto kernel_limits      = reserved::compute_kernel_limits(f, shared_mem_bytes, l0_sync || l1_sync);

    // For implicit widths, use sizes suggested by CUDA occupancy calculator
    if (l2_size == 0)
    {
      // Maximum occupancy without exceeding limits
      l2_size = ::std::min(kernel_limits.max_block_size, kernel_limits.block_size_limit);
    }
    else
    {
      if (int(l2_size) > kernel_limits.block_size_limit)
      {
        fprintf(stderr,
                "Unsatisfiable spec: Maximum block size %d threads, requested %zu (level 2)\n",
                kernel_limits.block_size_limit,
                l2_size);
        abort();
      }
    }

    if (l1_size == 0)
    {
      l1_size = kernel_limits.min_grid_size;
    }

    if (l0_size == 0)
    {
      l0_size = ndevs;
    }

    // Enforce the resource limits in the number of threads per block
    _CCCL_ASSERT(int(l2_size) <= kernel_limits.block_size_limit, "invalid level 2 size");
    _CCCL_ASSERT(int(l0_size) <= ndevs, "invalid level 0 size");

    /* Merge blocks and devices */
    this->add_level({::std::make_pair(hw_scope::device, l0_size)});
    this->set_level_mem(0, size_t(p.get_mem(0)));
    this->set_level_sync(0, l0_sync);

    this->add_level({::std::make_pair(hw_scope::block, l1_size)});
    this->set_level_mem(1, size_t(p.get_mem(1)));
    this->set_level_sync(1, l1_sync);

    this->add_level({::std::make_pair(hw_scope::thread, l2_size)});
    this->set_level_mem(2, size_t(p.get_mem(2)));
    this->set_level_sync(2, l2_sync);
  }
  else
  {
    static_assert(pdepth == 3);
  }
}
} // end namespace cuda::experimental::stf
