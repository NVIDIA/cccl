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
// Per-block execution shape; the dispatch picks `cluster_blocks` and the
// dynamic shared-memory tile capacity at runtime.
struct cluster_topk_policy
{
  int threads_per_block;
  int unroll_factor;
  int pipeline_stages;
  int chunk_bytes;
  int load_align_bytes;
  int bits_per_pass;
};

inline constexpr cluster_topk_policy default_policy{
  /*threads_per_block=*/256,
  /*unroll_factor=*/8,
  /*pipeline_stages=*/3,
  /*chunk_bytes=*/8 * 1024,
  /*load_align_bytes=*/128,
  /*bits_per_pass=*/8,
};
static_assert(default_policy.chunk_bytes % default_policy.load_align_bytes == 0);

// Default selector: one policy for every cluster-capable architecture
// (SM 9.0+). `unroll_factor` only controls ILP in runtime-sized chunk loops;
// it no longer defines the tile size. New tunings can be wired in by passing a
// custom selector to `dispatch` without changing the kernel signatures.
struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> cluster_topk_policy
  {
    return default_policy;
  }
};
} // namespace detail::batched_topk_cluster

CUB_NAMESPACE_END
