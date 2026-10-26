// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Private GroupMerge defaults: supported thread/IPT ranges and settings for each sorting launch.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_namespace.cuh>

#include <cuda/__device/compute_capability.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_capacity
{
// Settings for one launch; min/max_threads describe cooperation per segment, not block size.
struct group_merge_policy
{
  int block_threads;
  int min_threads;
  int max_threads;
  int min_ipt;
  int max_ipt;
  int blocks_per_sm;     // Grid-size multiplier: launch SM count * blocks_per_sm blocks.
  int min_blocks_per_sm; // __launch_bounds__ target for compiler register allocation, not the grid size.
  // Compare all settings when matching policies.
  _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const group_merge_policy& a, const group_merge_policy& b)
  {
    return a.block_threads == b.block_threads && a.min_threads == b.min_threads && a.max_threads == b.max_threads
        && a.min_ipt == b.min_ipt && a.max_ipt == b.max_ipt && a.blocks_per_sm == b.blocks_per_sm
        && a.min_blocks_per_sm == b.min_blocks_per_sm;
  }
};

// Bundle the three sorting launches: each member supplies a separate group_merge_policy
// for the warp/sub-warp, small-block, or large-block kernel. The dispatcher creates one entry per member.
struct group_merge_launch_policy
{
  group_merge_policy warp;
  group_merge_policy small;
  group_merge_policy large;
  // Compare the complete three-launch policy.
  _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const group_merge_launch_policy& a, const group_merge_launch_policy& b)
  {
    return a.warp == b.warp && a.small == b.small && a.large == b.large;
  }
};

// One implementation choice: three GroupMerge launches. Architecture-specific
// tuning values belong in this selector; other launch compositions are independent.
// Currently ignores compute capability. Keys wider than 8 bytes are intentionally
// unsupported until architecture-specific policies are added.
// TODO: Account for each architecture's shared-memory limit when enabling wider
// keys, while keeping the concatenated entry capacities strictly increasing.
template <typename Key>
struct group_merge_policy_selector
{
  // Select internal defaults by key width; architecture-specific values can be added here.
  _CCCL_HOST_DEVICE_API constexpr group_merge_launch_policy operator()(::cuda::compute_capability) const
  {
    constexpr int large_threads = sizeof(Key) <= 4 ? 1024 : 512;
    return {
      {256, 1, 32, 9, 15, 128, 1}, {256, 64, 256, 9, 15, 128, 1}, {large_threads, 512, large_threads, 9, 15, 2, 1}};
  }
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
