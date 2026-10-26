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
  int blocks_per_sm; // Grid-size multiplier: launch SM count * blocks_per_sm blocks.
  int min_blocks_per_sm; // __launch_bounds__ target for compiler register allocation, not the grid size.
};

// Select a tuning architecture, not a policy object. SM90 settings currently cover
// SM90 and newer; zero marks unsupported older targets. Add newer tuning tiers here.
struct group_merge_tuning_arch_selector
{
  _CCCL_HOST_DEVICE_API constexpr int operator()(::cuda::compute_capability cc) const
  {
    return cc >= ::cuda::compute_capability{9, 0} ? 90 : 0;
  }
};

// Each entry supplies its own reusable getter, parameterized by tuning architecture.
// The current values form the SM90 baseline; individual getters can add newer tunings.
template <int TuningArch>
struct group_merge_warp_policy_getter
{
  static_assert(TuningArch >= 90);
  _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const
  {
    return {256, 1, 32, 9, 15, 128, 1};
  }
};

template <int TuningArch>
struct group_merge_small_policy_getter
{
  static_assert(TuningArch >= 90);
  _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const
  {
    return {256, 64, 256, 9, 15, 128, 1};
  }
};

// Limit cooperating threads for larger keys to bound shared-memory scratch.
// Keys wider than 8 bytes need architecture-specific policies before being enabled.
template <int TuningArch, typename Key>
struct group_merge_large_policy_getter
{
  static_assert(TuningArch >= 90);
  _CCCL_HOST_DEVICE_API constexpr group_merge_policy operator()() const
  {
    constexpr int threads = sizeof(Key) <= 4 ? 1024 : 512;
    return {threads, 512, threads, 9, 15, 2, 1};
  }
};
} // namespace detail::segmented_sort_capacity

CUB_NAMESPACE_END
