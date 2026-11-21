// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_POLICY_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
#  include <cub/block/block_store.cuh>

#  include <cuda/experimental/mgmn/__algorithm/common.h>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
namespace __multiway
{
inline constexpr int mway_max_k_limit = 32; // one warp lane per input array

struct mway_policy
{
  int threads_per_block;
  int items_per_thread;
  CUB_NS_QUALIFIER::BlockStoreAlgorithm store_algorithm;

  [[nodiscard]] constexpr int items_per_tile() const
  {
    return threads_per_block * items_per_thread;
  }
};

//! Spell out a K-way merge policy explicitly. The agent uses one shared-memory buffer of
//! `items_per_tile + 1` keys (plus one 2-byte index per key for pairs) and stages every merge-tree
//! round through a per-thread register window of items_per_thread keys (see agent_mway_merge.cuh),
//! so the tile is bounded by two things: `items_per_tile * sizeof(KeyT)` under the default 48 KiB
//! static shared-memory limit (this version does not opt in to the larger dynamic limit the 2-way
//! agent's bulk-copy path uses), and the register window, which grows with items_per_thread x
//! sizeof(KeyT) and caps the blocks per SM before shared memory does for 4-byte keys.
template <int ThreadsPerBlock,
          int ItemsPerThread,
          CUB_NS_QUALIFIER::BlockStoreAlgorithm StoreAlgorithm = CUB_NS_QUALIFIER::BLOCK_STORE_WARP_TRANSPOSE>
struct static_mway_policy
{
  static constexpr mway_policy value{ThreadsPerBlock, ItemsPerThread, StoreAlgorithm};
};

//! Default policy for the standalone K-way merge: modest tile size, well under the 48 KiB static
//! shared-memory limit for any key type this repo tests (up to 8 bytes). The mergesort picks its
//! own, larger tiles (default_mway_sort_merge_policy in mway_sort.cuh).
using default_mway_policy = static_mway_policy<256, 8>;
} // namespace __multiway
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILER(NVCC)

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_POLICY_H
