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
namespace detail::segmented_topk
{
template <typename KeyT, typename ValueT, typename SegmentSizeT, ::cuda::std::int64_t MaxK>
struct policy_hub
{
  struct Policy900 : ChainedPolicy<900, Policy900, Policy900>
  {
    // Policies selected based on optimal performance for different segment sizes
    // The list below will be checked to determine if each policy can support the one-worker-per-segment approach
    // within available shared memory limits. Policies must be ordered by decreasing segment size
    using worker_per_segment_policies =
      ::cuda::std::tuple<AgentBatchedTopKWorkerPerSegmentPolicy<256, 64>,
                         AgentBatchedTopKWorkerPerSegmentPolicy<256, 32>,
                         AgentBatchedTopKWorkerPerSegmentPolicy<256, 16>,
                         AgentBatchedTopKWorkerPerSegmentPolicy<256, 8>,
                         AgentBatchedTopKWorkerPerSegmentPolicy<256, 4>,
                         AgentBatchedTopKWorkerPerSegmentPolicy<128, 2>>;
  };

  using max_policy = Policy900;
};
} // namespace detail::segmented_topk

CUB_NAMESPACE_END
