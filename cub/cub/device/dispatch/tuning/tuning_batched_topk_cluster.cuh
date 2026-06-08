// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/std/inplace_vector>

CUB_NAMESPACE_BEGIN
namespace detail::batched_topk_cluster
{
inline constexpr int max_launch_configs = 19;

struct cluster_topk_launch_config
{
  int cluster_blocks;
  // Total per-block shared memory budget, including the agent's static shared memory.
  int total_smem_bytes;
};

// Per-block execution shape; the dispatch picks `cluster_blocks` and the
// dynamic shared-memory block_tile capacity at runtime.
struct cluster_topk_policy
{
  int threads_per_block;
  int unroll_factor;
  int pipeline_stages;
  int chunk_bytes;
  int load_align_bytes;
  int bits_per_pass;
  // Minimum number of CTAs per SM passed to the dynamic-cluster kernel's launch bounds. Defaults to 1: most cluster
  // configurations already have an occupancy of 1 due to their shared-memory usage, so allowing the compiler to
  // optimize for higher occupancy (fewer registers per thread) provides no benefit.
  int min_blocks_per_sm;
  ::cuda::std::inplace_vector<cluster_topk_launch_config, max_launch_configs> launch_configs;
};

template <typename... Configs>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_launch_configs(Configs... configs)
  -> ::cuda::std::inplace_vector<cluster_topk_launch_config, max_launch_configs>
{
  ::cuda::std::inplace_vector<cluster_topk_launch_config, max_launch_configs> result;
  (result.emplace_back(configs), ...);
  return result;
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
make_policy(::cuda::std::inplace_vector<cluster_topk_launch_config, max_launch_configs> launch_configs)
  -> cluster_topk_policy
{
  return cluster_topk_policy{
    /*threads_per_block=*/512,
    /*unroll_factor=*/0,
    /*pipeline_stages=*/3,
    /*chunk_bytes=*/16 * 1024,
    /*load_align_bytes=*/128,
    /*bits_per_pass=*/11,
    /*min_blocks_per_sm=*/1,
    launch_configs};
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm90_100_110_policy() -> cluster_topk_policy
{
  return make_policy(make_launch_configs(
    cluster_topk_launch_config{2, 31 * 1024}, // 62 KiB
    // cluster_topk_launch_config{4, 31 * 1024}, // 124 KiB
    cluster_topk_launch_config{2, 63 * 1024}, // 126 KiB
    cluster_topk_launch_config{2, 99 * 1024}, // 198 KiB
    // cluster_topk_launch_config{8, 31 * 1024}, // 248 KiB
    // cluster_topk_launch_config{4, 63 * 1024}, // 252 KiB
    cluster_topk_launch_config{2, 131 * 1024}, // 262 KiB
    cluster_topk_launch_config{2, 163 * 1024}, // 326 KiB
    cluster_topk_launch_config{2, 195 * 1024}, // 390 KiB
    // cluster_topk_launch_config{4, 99 * 1024}, // 396 KiB
    cluster_topk_launch_config{2, 227 * 1024}, // 454 KiB
    // cluster_topk_launch_config{8, 63 * 1024}, // 504 KiB
    cluster_topk_launch_config{4, 131 * 1024}, // 524 KiB
    cluster_topk_launch_config{4, 163 * 1024}, // 652 KiB
    cluster_topk_launch_config{4, 195 * 1024}, // 780 KiB
    // cluster_topk_launch_config{8, 99 * 1024}, // 792 KiB
    cluster_topk_launch_config{4, 227 * 1024}, // 908 KiB
    cluster_topk_launch_config{8, 131 * 1024}, // 1048 KiB
    cluster_topk_launch_config{8, 163 * 1024}, // 1304 KiB
    cluster_topk_launch_config{8, 195 * 1024}, // 1560 KiB
    cluster_topk_launch_config{8, 227 * 1024}, // 1816 KiB
    // cluster_topk_launch_config{16, 131 * 1024}, // 2096 KiB
    // cluster_topk_launch_config{16, 163 * 1024}, // 2608 KiB
    // cluster_topk_launch_config{16, 195 * 1024}, // 3120 KiB
    cluster_topk_launch_config{16, 227 * 1024} // 3632 KiB
    ));
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto sm120_policy() -> cluster_topk_policy
{
  return make_policy(make_launch_configs(
    cluster_topk_launch_config{2, 31 * 1024}, // 62 KiB
    // cluster_topk_launch_config{4, 31 * 1024}, // 124 KiB
    cluster_topk_launch_config{2, 63 * 1024}, // 126 KiB
    cluster_topk_launch_config{2, 99 * 1024}, // 198 KiB
    // cluster_topk_launch_config{8, 31 * 1024}, // 248 KiB
    cluster_topk_launch_config{4, 63 * 1024}, // 252 KiB
    cluster_topk_launch_config{4, 99 * 1024}, // 396 KiB
    cluster_topk_launch_config{8, 63 * 1024}, // 504 KiB
    cluster_topk_launch_config{8, 99 * 1024} // 792 KiB
    ));
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto is_valid_policy(cluster_topk_policy policy) -> bool
{
  return policy.chunk_bytes % policy.load_align_bytes == 0 && policy.launch_configs.size() <= max_launch_configs;
}

static_assert(is_valid_policy(sm90_100_110_policy()));
static_assert(is_valid_policy(sm120_policy()));
// Default selector for cluster-capable architectures (SM 9.0+).
// `unroll_factor` only controls ILP in runtime-sized chunk loops; it no longer
// defines the block_tile size. The launch config table is consumed at runtime, so
// selectors can vary it by architecture without forcing static dispatch over
// all cluster/SMEM combinations.
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> cluster_topk_policy
  {
    if (cc >= ::cuda::compute_capability{12, 0})
    {
      return sm120_policy();
    }

    return sm90_100_110_policy();
  }
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
