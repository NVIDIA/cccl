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

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_load.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/__algorithm/clamp.h>

CUB_NAMESPACE_BEGIN
namespace detail::segmented_topk
{
template <typename KeyT, typename ValueT, typename SegmentSizeT, ::cuda::std::int64_t MaxK>
struct policy_hub
{
  struct Policy900 : ChainedPolicy<900, Policy900, Policy900>
  {
    // This is a sequence of policies that turned out to hit some optimum performance for a certain segment size
    // This list will be examined whether they can be used in one-worker-per-segment approach without exceeding shared
    // memory. The sequence must be ordered by segment size in descending order
    using worker_per_segment_policies =
      ::cuda::std::tuple<AgentSegmentedTopkWorkerPerSegmentPolicy<512, 8>,
                         AgentSegmentedTopkWorkerPerSegmentPolicy<512, 4>,
                         AgentSegmentedTopkWorkerPerSegmentPolicy<256, 8>,
                         AgentSegmentedTopkWorkerPerSegmentPolicy<128, 2>>;
  };

  using max_policy = Policy900;
};
} // namespace detail::segmented_topk

CUB_NAMESPACE_END
