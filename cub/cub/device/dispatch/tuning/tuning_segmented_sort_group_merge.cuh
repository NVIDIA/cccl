// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private partition launch settings and GroupMerge thread/IPT ranges and launch defaults.

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

namespace detail::segmented_sort_lrb
{
// Count and scatter use the same block size, with one input segment per thread.
struct partition_policy
{
  int threads_per_block = 256;
};

// Settings for one GroupMerge sorting launch.
struct group_merge_policy
{
  int threads_per_block;
  int min_threads_per_segment;
  int max_threads_per_segment;
  int min_items_per_thread;
  int max_items_per_thread;
  int grid_size_multiplier; // Launch SM count * grid_size_multiplier blocks.
  int min_resident_blocks_per_sm; // Compiler launch-bounds target, not the grid size.

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const group_merge_policy& lhs, const group_merge_policy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.min_threads_per_segment == rhs.min_threads_per_segment
        && lhs.max_threads_per_segment == rhs.max_threads_per_segment
        && lhs.min_items_per_thread == rhs.min_items_per_thread && lhs.max_items_per_thread == rhs.max_items_per_thread
        && lhs.grid_size_multiplier == rhs.grid_size_multiplier
        && lhs.min_resident_blocks_per_sm == rhs.min_resident_blocks_per_sm;
  }
};

struct group_merge_policies
{
  group_merge_policy warp;
  group_merge_policy small_block;
  group_merge_policy large;
};

// Internal defaults only; no public tuning customization point is introduced.
struct lrb_policy
{
  partition_policy partition;
  group_merge_policies group_merge{};

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const lrb_policy& lhs, const lrb_policy& rhs) noexcept
  {
    return lhs.partition.threads_per_block == rhs.partition.threads_per_block
        && lhs.group_merge.warp == rhs.group_merge.warp && lhs.group_merge.small_block == rhs.group_merge.small_block
        && lhs.group_merge.large == rhs.group_merge.large;
  }
};

template <typename Key>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr lrb_policy operator()(::cuda::compute_capability) const noexcept
  {
    constexpr int large_threads_per_block = sizeof(Key) <= 4 ? 1024 : 512;
    lrb_policy policy{};
    auto& warp                      = policy.group_merge.warp;
    warp.threads_per_block          = 256;
    warp.min_threads_per_segment    = 1;
    warp.max_threads_per_segment    = 32;
    warp.min_items_per_thread       = 9;
    warp.max_items_per_thread       = 15;
    warp.grid_size_multiplier       = 128;
    warp.min_resident_blocks_per_sm = 1;

    auto& small_block                   = policy.group_merge.small_block;
    small_block                         = warp;
    small_block.min_threads_per_segment = 64;
    small_block.max_threads_per_segment = 256;

    auto& large                   = policy.group_merge.large;
    large                         = small_block;
    large.threads_per_block       = large_threads_per_block;
    large.min_threads_per_segment = 512;
    large.max_threads_per_segment = large_threads_per_block;
    large.grid_size_multiplier    = 2;
    return policy;
  }
};

// Generic partition-only callers use the same fixed default.
struct partition_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr partition_policy operator()(::cuda::compute_capability) const noexcept
  {
    return {};
  }
};

template <typename PolicySelector>
struct partition_subpolicy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr partition_policy operator()(::cuda::compute_capability cc) const noexcept
  {
    return PolicySelector{}(cc).partition;
  }
};

// Project one launch's policy on both the host and device, from the same parent selector.
template <typename PolicySelector, group_merge_policy group_merge_policies::* Member>
struct group_merge_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr group_merge_policy
  operator()(::cuda::compute_capability cc) const noexcept
  {
    return PolicySelector{}(cc).group_merge.*Member;
  }

  template <typename PolicyGetter>
  struct getter
  {
    [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const noexcept
    {
      return PolicyGetter{}().group_merge.*Member;
    }
  };
};
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END
