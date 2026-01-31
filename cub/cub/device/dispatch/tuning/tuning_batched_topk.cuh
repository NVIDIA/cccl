// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_batched_topk.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN
namespace detail::batched_topk
{
template <typename KeyT, typename ValueT, typename SegmentSizeT, ::cuda::std::int64_t MaxK>
struct policy_hub
{
  struct Policy900 : ChainedPolicy<900, Policy900, Policy900>
  {
    static constexpr BlockLoadAlgorithm default_load_alg   = BLOCK_LOAD_WARP_TRANSPOSE;
    static constexpr BlockStoreAlgorithm default_store_alg = BLOCK_STORE_WARP_TRANSPOSE;

    // The list below will be checked to determine if each policy can support the one-worker-per-segment approach
    // within available shared memory limits. The first policy that fits SMEM is taken. Policies must be ordered by
    // decreasing segment size.
    using worker_per_segment_policies =
      ::cuda::std::tuple<agent_batched_topk_worker_per_segment_policy<256, 64, default_load_alg, default_store_alg>,
                         agent_batched_topk_worker_per_segment_policy<256, 32, default_load_alg, default_store_alg>,
                         agent_batched_topk_worker_per_segment_policy<256, 16, default_load_alg, default_store_alg>,
                         agent_batched_topk_worker_per_segment_policy<256, 8, default_load_alg, default_store_alg>,
                         agent_batched_topk_worker_per_segment_policy<256, 4, default_load_alg, default_store_alg>,
                         agent_batched_topk_worker_per_segment_policy<128, 2, default_load_alg, default_store_alg>>;
  };

  using max_policy = Policy900;
};
} // namespace detail::batched_topk

CUB_NAMESPACE_END
