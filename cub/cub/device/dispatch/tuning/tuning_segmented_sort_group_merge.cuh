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
  int block_threads = 256;
};

// Settings for one launch; min/max_threads describe cooperation per segment, not block size.
struct group_merge_policy
{
  int block_threads;
  int min_threads;
  int max_threads;
  int min_ipt;
  int max_ipt;
  int grid_size_multiplier; // Launch SM count * grid_size_multiplier blocks.
  int min_blocks_per_sm; // Compiler launch-bounds target, not the grid size.

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const group_merge_policy& lhs, const group_merge_policy& rhs) noexcept
  {
    return lhs.block_threads == rhs.block_threads && lhs.min_threads == rhs.min_threads
        && lhs.max_threads == rhs.max_threads && lhs.min_ipt == rhs.min_ipt && lhs.max_ipt == rhs.max_ipt
        && lhs.grid_size_multiplier == rhs.grid_size_multiplier && lhs.min_blocks_per_sm == rhs.min_blocks_per_sm;
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
    return lhs.partition.block_threads == rhs.partition.block_threads && lhs.group_merge.warp == rhs.group_merge.warp
        && lhs.group_merge.small_block == rhs.group_merge.small_block && lhs.group_merge.large == rhs.group_merge.large;
  }
};

template <typename Key>
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr lrb_policy operator()(::cuda::compute_capability) const noexcept
  {
    constexpr int threads = sizeof(Key) <= 4 ? 1024 : 512;
    return {{256}, {{256, 1, 32, 9, 15, 128, 1}, {256, 64, 256, 9, 15, 128, 1}, {threads, 512, threads, 9, 15, 2, 1}}};
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
