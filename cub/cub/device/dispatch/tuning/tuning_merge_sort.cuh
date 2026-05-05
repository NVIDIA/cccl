// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

template <int BlockThreads,
          int ItemsPerThread                      = 1,
          cub::BlockLoadAlgorithm LoadAlgorithm   = cub::BLOCK_LOAD_DIRECT,
          cub::CacheLoadModifier LoadModifier     = cub::LOAD_LDG,
          cub::BlockStoreAlgorithm StoreAlgorithm = cub::BLOCK_STORE_DIRECT>
struct AgentMergeSortPolicy
{
  static constexpr int BLOCK_THREADS    = BlockThreads;
  static constexpr int ITEMS_PER_THREAD = ItemsPerThread;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;

  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = LoadAlgorithm;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = LoadModifier;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = StoreAlgorithm;
};

namespace detail::merge_sort
{
// TODO(bgruber): drop in CCCL 4.0 when we remove all public CUB dispatchers
template <typename KeyIteratorT>
struct policy_hub
{
  using KeyT = it_value_t<KeyIteratorT>;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(11),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };

  // NVBug 3384810
#if defined(_NVHPC_CUDA)
  using Policy520 = Policy500;
#else
  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<512,
                           Nominal4BItemsToItems<KeyT>(15),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };
#endif

  struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(17),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_DEFAULT,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy600;
};

struct merge_sort_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;

  [[nodiscard]] _CCCL_API constexpr int items_per_tile() const
  {
    return block_threads * items_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const merge_sort_policy& lhs, const merge_sort_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const merge_sort_policy& lhs, const merge_sort_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const merge_sort_policy& p)
  {
    return os << "merge_sort_policy { .block_threads = " << p.block_threads
              << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
              << ", .load_modifier = " << p.load_modifier << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept merge_sort_policy_selector = policy_selector<T, merge_sort_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability) const -> merge_sort_policy
  {
    // from SM60
    return merge_sort_policy{
      256,
      detail::nominal_4B_items_to_items(17, key_size),
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      BLOCK_STORE_WARP_TRANSPOSE};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(merge_sort_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyIteratorT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> merge_sort_policy
  {
    return policy_selector{int{sizeof(it_value_t<KeyIteratorT>)}}(cc);
  }
};

// TODO(bgruber): remove in CCCL 4.0
template <typename PolicyHub>
struct policy_selector_from_hub
{
  // this is only called in device code, so we can ignore the cc parameter
  _CCCL_HOST_DEVICE constexpr auto operator()(::cuda::compute_capability) const -> merge_sort_policy
  {
    using ap = typename PolicyHub::MaxPolicy::ActivePolicy;
    using mp = typename ap::MergeSortPolicy;
    return merge_sort_policy{
      mp::BLOCK_THREADS, mp::ITEMS_PER_THREAD, mp::LOAD_ALGORITHM, mp::LOAD_MODIFIER, mp::STORE_ALGORITHM};
  }
};
} // namespace detail::merge_sort

CUB_NAMESPACE_END
