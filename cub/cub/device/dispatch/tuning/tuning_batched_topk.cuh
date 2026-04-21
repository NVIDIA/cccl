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

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN
namespace detail::batched_topk
{
struct worker_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  BlockStoreAlgorithm store_algorithm;

  _CCCL_API constexpr friend bool operator==(const worker_policy& lhs, const worker_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.store_algorithm == rhs.store_algorithm;
  }

  _CCCL_API constexpr friend bool operator!=(const worker_policy& lhs, const worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const worker_policy& p)
  {
    return os
        << "worker_policy { .block_threads = " << p.block_threads << ", .items_per_thread = " << p.items_per_thread
        << ", .load_algorithm = " << p.load_algorithm << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct batched_topk_policy
{
  // The list of per-segment agent policies is ordered by decreasing tile size. At compile time, the smallest policy
  // whose tile size still covers the upper bound of the segment size is selected.
  ::cuda::std::array<worker_policy, 6> worker_per_segment_policies;

  _CCCL_API constexpr friend bool operator==(const batched_topk_policy& lhs, const batched_topk_policy& rhs)
  {
    return lhs.worker_per_segment_policies == rhs.worker_per_segment_policies;
  }

  _CCCL_API constexpr friend bool operator!=(const batched_topk_policy& lhs, const batched_topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const batched_topk_policy& p)
  {
    os << "batched_topk_policy { .worker_per_segment_policies = { ";
    for (::cuda::std::size_t i = 0; i < p.worker_per_segment_policies.size(); ++i)
    {
      if (i != 0)
      {
        os << ", ";
      }
      os << p.worker_per_segment_policies[i];
    }
    return os << " } }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept batched_topk_policy_selector = policy_selector<T, batched_topk_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> batched_topk_policy
  {
    constexpr auto load_alg  = BLOCK_LOAD_WARP_TRANSPOSE;
    constexpr auto store_alg = BLOCK_STORE_WARP_TRANSPOSE;
    return batched_topk_policy{{{
      worker_policy{256, 64, load_alg, store_alg},
      worker_policy{256, 32, load_alg, store_alg},
      worker_policy{256, 16, load_alg, store_alg},
      worker_policy{256, 8, load_alg, store_alg},
      worker_policy{256, 4, load_alg, store_alg},
      worker_policy{128, 2, load_alg, store_alg},
    }}};
  }
};

template <typename KeyT, typename ValueT, typename SegmentSizeT, ::cuda::std::int64_t MaxK>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> batched_topk_policy
  {
    return policy_selector{}(arch);
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(batched_topk_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::batched_topk

CUB_NAMESPACE_END
