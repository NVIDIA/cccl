//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Tree structure to describe the parent/child hierarchy of stackable context nodes

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <algorithm>
#include <vector>

#include "cuda/experimental/__stf/utility/nvtx.cuh"

namespace cuda::experimental::stf
{
//! @brief Tree of integer offsets describing the hierarchy of stackable context nodes.
//!
//! Every node is identified by an integer offset.  The class maintains
//! parent/child relationships and a free-list so that offsets can be reused
//! after a sub-context is destroyed.  The pool grows automatically using the
//! provided growth factor when all slots are exhausted.
class node_hierarchy
{
public:
  node_hierarchy(int initial_size  = default_initial_size,
                 size_t growth_num = default_growth_numerator,
                 size_t growth_den = default_growth_denominator)
      : growth_factor_numerator(growth_num)
      , growth_factor_denominator(growth_den)
  {
    parent.resize(initial_size);
    children.resize(initial_size);
    free_list.reserve(initial_size);

    for (int i = 0; i < initial_size; i++)
    {
      free_list.push_back(initial_size - 1 - i);
    }
  }

  void grow()
  {
    int old_size = static_cast<int>(parent.size());
    int new_size =
      ::std::max(old_size + 1, static_cast<int>(old_size * growth_factor_numerator / growth_factor_denominator));
    parent.resize(new_size);
    children.resize(new_size);
    for (int i = new_size - 1; i >= old_size; i--)
    {
      free_list.push_back(i);
    }
  }

  int get_avail_entry()
  {
    if (free_list.empty())
    {
      grow();
    }

    _CCCL_ASSERT(!free_list.empty(), "no slot available");

    int result = free_list.back();
    free_list.pop_back();

    _CCCL_ASSERT(children[result].empty(), "invalid state");
    parent[result] = -1;

    return result;
  }

  //! Return a node offset to the free-list so it can be reused.
  void discard_node(int offset)
  {
    nvtx_range r("discard_node");

    int p = parent[offset];
    if (p != -1)
    {
      auto& parent_children = children[p];
      auto it               = ::std::find(parent_children.begin(), parent_children.end(), offset);
      _CCCL_ASSERT(it != parent_children.end(), "invalid hierarchy state");
      parent_children.erase(it);
    }

    children[offset].clear();
    parent[offset] = -1;

    free_list.push_back(offset);
  }

  void set_parent(int parent_offset, int child_offset)
  {
    parent[child_offset] = parent_offset;
    children[parent_offset].push_back(child_offset);
  }

  int get_parent(int offset) const
  {
    _CCCL_ASSERT(offset < int(parent.size()), "node offset exceeds parent array size");
    return parent[offset];
  }

  const auto& get_children(int offset) const
  {
    _CCCL_ASSERT(offset < int(children.size()), "node offset exceeds children array size");
    return children[offset];
  }

  size_t depth(int offset) const
  {
    int p = get_parent(offset);
    if (p == -1)
    {
      return 0;
    }

    return 1 + depth(p);
  }

  static constexpr int default_initial_size          = 16;
  static constexpr size_t default_growth_numerator   = 3;
  static constexpr size_t default_growth_denominator = 2;

private:
  size_t growth_factor_numerator;
  size_t growth_factor_denominator;

  //! Offset of the node's parent : -1 if none.  Only valid for entries not in free-list.
  ::std::vector<int> parent;

  //! If a node has children, indicate their offset here
  ::std::vector<::std::vector<int>> children;

  //! Available offsets to create new nodes
  ::std::vector<int> free_list;
};
} // namespace cuda::experimental::stf
