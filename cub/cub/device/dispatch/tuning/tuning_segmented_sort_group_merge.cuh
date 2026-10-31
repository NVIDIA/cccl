// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  int blocks_per_sm; // Grid-size multiplier: launch SM count * blocks_per_sm blocks.
  int min_blocks_per_sm; // __launch_bounds__ target for compiler register allocation, not the grid size.
};

// Preserve the compiled compute capability for host-side setup. Dispatch rejects targets below SM90.
struct group_merge_tuning_arch_selector
{
  _CCCL_HOST_DEVICE_API constexpr ::cuda::compute_capability operator()(::cuda::compute_capability cc) const
  {
    return cc;
  }
};

// Independent selectors supply the current SM90+ defaults for each sorting launch.
// Older targets can compile these kernels, but the host dispatcher does not launch them.
struct group_merge_warp_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()(::cuda::compute_capability) const
  {
    return {256, 1, 32, 9, 15, 128, 1};
  }
};

struct group_merge_small_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()(::cuda::compute_capability) const
  {
    return {256, 64, 256, 9, 15, 128, 1};
  }
};

// Limit cooperating threads for larger keys to bound shared-memory scratch.
template <typename Key>
struct group_merge_large_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()(::cuda::compute_capability) const
  {
    constexpr int threads = sizeof(Key) <= 4 ? 1024 : 512;
    return {threads, 512, threads, 9, 15, 2, 1};
  }
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
