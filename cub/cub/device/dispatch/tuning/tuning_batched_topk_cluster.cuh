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

CUB_NAMESPACE_BEGIN
namespace detail::batched_topk_cluster
{
// Per-block execution shape. The dispatch picks the cluster size and the dynamic shared-memory block_tile capacity at
// runtime (occupancy/wave-aware), so the policy carries only the per-block tuning knobs.
struct cluster_topk_policy
{
  int threads_per_block;
  int histogram_items_per_thread;
  int pipeline_stages;
  int chunk_bytes;
  int load_align_bytes;
  int bits_per_pass;
  int min_blocks_per_sm;
  int tie_break_items_per_thread;
};

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_policy() -> cluster_topk_policy
{
  return cluster_topk_policy{
    /*threads_per_block=*/512,
    /*histogram_items_per_thread=*/8,
    /*pipeline_stages=*/8,
    /*chunk_bytes=*/16 * 1024,
    /*load_align_bytes=*/128,
    /*bits_per_pass=*/11,
    /*min_blocks_per_sm=*/1,
    /*tie_break_items_per_thread=*/8};
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto is_valid_policy(cluster_topk_policy policy) -> bool
{
  return policy.chunk_bytes % policy.load_align_bytes == 0;
}

static_assert(is_valid_policy(make_policy()));
// Default selector for cluster-capable architectures (SM 9.0+). The tuning is currently identical across CCs.
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> cluster_topk_policy
  {
    return make_policy();
  }
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
