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

#include <cub/agent/agent_segmented_radix_sort.cuh>
#include <cub/agent/agent_sub_warp_merge_sort.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/concepts>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort
{
struct segmented_radix_sort_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  RadixRankAlgorithm rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int radix_bits;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const segmented_radix_sort_policy& lhs, const segmented_radix_sort_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.rank_algorithm == rhs.rank_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const segmented_radix_sort_policy& lhs, const segmented_radix_sort_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_radix_sort_policy& p)
  {
    return os
        << "segmented_radix_sort_policy { .block_threads = " << p.block_threads
        << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .rank_algorithm = " << p.rank_algorithm
        << ", .scan_algorithm = " << p.scan_algorithm << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // _CCCL_HOSTED()
};

struct sub_warp_merge_sort_policy
{
  int block_threads;
  int warp_threads;
  int items_per_thread;
  WarpLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  WarpStoreAlgorithm store_algorithm;

  [[nodiscard]] _CCCL_API constexpr int segments_per_block() const
  {
    return block_threads / warp_threads;
  }

  [[nodiscard]] _CCCL_API constexpr int items_per_tile() const
  {
    return warp_threads * items_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const sub_warp_merge_sort_policy& lhs, const sub_warp_merge_sort_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.warp_threads == rhs.warp_threads
        && lhs.items_per_thread == rhs.items_per_thread && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.store_algorithm == rhs.store_algorithm;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const sub_warp_merge_sort_policy& lhs, const sub_warp_merge_sort_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const sub_warp_merge_sort_policy& p)
  {
    return os
        << "sub_warp_merge_sort_policy { .block_threads = " << p.block_threads << ", .warp_threads = " << p.warp_threads
        << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

struct segmented_sort_policy
{
  segmented_radix_sort_policy large_segment;
  sub_warp_merge_sort_policy small_segment;
  sub_warp_merge_sort_policy medium_segment;
  int partitioning_threshold;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const segmented_sort_policy& lhs, const segmented_sort_policy& rhs)
  {
    return lhs.large_segment == rhs.large_segment && lhs.small_segment == rhs.small_segment
        && lhs.medium_segment == rhs.medium_segment && lhs.partitioning_threshold == rhs.partitioning_threshold;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const segmented_sort_policy& lhs, const segmented_sort_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_sort_policy& p)
  {
    return os << "segmented_sort_policy { .large_segment = " << p.large_segment
              << ", .small_segment = " << p.small_segment << ", .medium_segment = " << p.medium_segment
              << ", .partitioning_threshold = " << p.partitioning_threshold << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_sort_policy_selector = policy_selector<T, segmented_sort_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;
  int value_size;
  bool keys_only;

  _CCCL_API constexpr auto __dominant_size() const
  {
    return ::cuda::std::max(key_size, value_size);
  }

  _CCCL_API constexpr auto __make_scaled_segmented_radix_sort_policy(
    int nominal_4B_block_threads,
    int nominal_4B_items_per_thread,
    BlockLoadAlgorithm load_algorithm,
    CacheLoadModifier load_modifier,
    RadixRankAlgorithm rank_algorithm,
    BlockScanAlgorithm scan_algorithm,
    int radix_bits) const
  {
    const auto scaled = scale_reg_bound(nominal_4B_block_threads, nominal_4B_items_per_thread, __dominant_size());
    return segmented_radix_sort_policy{
      scaled.block_threads,
      scaled.items_per_thread,
      load_algorithm,
      load_modifier,
      rank_algorithm,
      scan_algorithm,
      radix_bits};
  }

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> segmented_sort_policy
  {
    const auto scale_items = [&](int nominal_4b_items_per_thread) {
      return nominal_4B_items_to_items(nominal_4b_items_per_thread, __dominant_size());
    };

    if (cc >= ::cuda::compute_capability{8, 6})
    {
      const bool large_items = __dominant_size() > 4;
      const int radix_bits   = key_size > 1 ? 6 : 4;
      const int small_itp    = scale_items(large_items ? 7 : 9);
      const int medium_itp   = scale_items(large_items ? 9 : 7);
      return segmented_sort_policy{
        __make_scaled_segmented_radix_sort_policy(
          256, 23, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        sub_warp_merge_sort_policy{
          256, large_items ? 8 : 2, small_itp, WARP_LOAD_TRANSPOSE, LOAD_LDG, WARP_STORE_DIRECT},
        sub_warp_merge_sort_policy{256, 16, medium_itp, WARP_LOAD_TRANSPOSE, LOAD_LDG, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      const int radix_bits = key_size > 1 ? 6 : 4;
      const int small_itp  = scale_items(9);
      const int medium_itp = scale_items(keys_only ? 7 : 11);
      return segmented_sort_policy{
        __make_scaled_segmented_radix_sort_policy(
          256, 23, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        sub_warp_merge_sort_policy{
          256, keys_only ? 4 : 2, small_itp, WARP_LOAD_TRANSPOSE, LOAD_DEFAULT, WARP_STORE_DIRECT},
        sub_warp_merge_sort_policy{256, 32, medium_itp, WARP_LOAD_TRANSPOSE, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{7, 0})
    {
      const int radix_bits = key_size > 1 ? 6 : 4;
      const int small_itp  = scale_items(7);
      const int medium_itp = scale_items(keys_only ? 11 : 7);
      return segmented_sort_policy{
        __make_scaled_segmented_radix_sort_policy(
          256, 19, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        sub_warp_merge_sort_policy{256, keys_only ? 4 : 8, small_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        sub_warp_merge_sort_policy{256, 32, medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{6, 2})
    {
      const int radix_bits       = key_size > 1 ? 5 : 4;
      const int small_medium_itp = scale_items(9);
      return segmented_sort_policy{
        __make_scaled_segmented_radix_sort_policy(
          256, 16, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, radix_bits),
        sub_warp_merge_sort_policy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        sub_warp_merge_sort_policy{256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{6, 1})
    {
      const int radix_bits       = key_size > 1 ? 6 : 4;
      const int small_medium_itp = scale_items(9);
      return segmented_sort_policy{
        __make_scaled_segmented_radix_sort_policy(
          256, 19, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        sub_warp_merge_sort_policy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        sub_warp_merge_sort_policy{256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      const int radix_bits       = key_size > 1 ? 6 : 4;
      const int small_medium_itp = scale_items(9);
      return segmented_sort_policy{
        __make_scaled_segmented_radix_sort_policy(
          256, 19, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, radix_bits),
        sub_warp_merge_sort_policy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        sub_warp_merge_sort_policy{256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    // default for SM50
    const int radix_bits       = key_size > 1 ? 6 : 4;
    const int small_medium_itp = scale_items(7);
    return segmented_sort_policy{
      __make_scaled_segmented_radix_sort_policy(
        256, 16, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, radix_bits),
      sub_warp_merge_sort_policy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
      sub_warp_merge_sort_policy{256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
      300};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_sort_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyT, typename ValueT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> segmented_sort_policy
  {
    return policy_selector{int{sizeof(KeyT)}, int{sizeof(ValueT)}, ::cuda::std::is_same_v<ValueT, NullType>}(cc);
  }
};

// TODO(bgruber): remove when we drop the CUB dispatchers in CCCL 4.0
template <typename PolicyT, typename = void>
struct SegmentedSortPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE SegmentedSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

// TODO(bgruber): remove when we drop the CUB dispatchers in CCCL 4.0
template <typename StaticPolicyT>
struct SegmentedSortPolicyWrapper<StaticPolicyT,
                                  ::cuda::std::void_t<typename StaticPolicyT::LargeSegmentPolicy,
                                                      typename StaticPolicyT::SmallSegmentPolicy,
                                                      typename StaticPolicyT::MediumSegmentPolicy>> : StaticPolicyT
{
  _CCCL_HOST_DEVICE SegmentedSortPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr int PartitioningThreshold()
  {
    return StaticPolicyT::PARTITIONING_THRESHOLD;
  }

  _CCCL_HOST_DEVICE static constexpr int LargeSegmentRadixBits()
  {
    return StaticPolicyT::LargeSegmentPolicy::RADIX_BITS;
  }

  _CCCL_HOST_DEVICE static constexpr int LargeSegmentBlockThreads()
  {
    return StaticPolicyT::LargeSegmentPolicy::BLOCK_THREADS;
  }

  _CCCL_HOST_DEVICE static constexpr int LargeSegmentItemsPerThread()
  {
    return StaticPolicyT::LargeSegmentPolicy::ITEMS_PER_THREAD;
  }

  _CCCL_HOST_DEVICE static constexpr int SmallSegmentBlockThreads()
  {
    return StaticPolicyT::SmallSegmentPolicy::BLOCK_THREADS;
  }

  _CCCL_HOST_DEVICE static constexpr int SegmentsPerSmallBlock()
  {
    return StaticPolicyT::SmallSegmentPolicy::SEGMENTS_PER_BLOCK;
  }

  _CCCL_HOST_DEVICE static constexpr int SegmentsPerMediumBlock()
  {
    return StaticPolicyT::MediumSegmentPolicy::SEGMENTS_PER_BLOCK;
  }

  _CCCL_HOST_DEVICE static constexpr int SmallPolicyItemsPerTile()
  {
    return StaticPolicyT::SmallSegmentPolicy::ITEMS_PER_TILE;
  }

  _CCCL_HOST_DEVICE static constexpr int MediumPolicyItemsPerTile()
  {
    return StaticPolicyT::MediumSegmentPolicy::ITEMS_PER_TILE;
  }

  _CCCL_HOST_DEVICE static constexpr CacheLoadModifier LargeSegmentLoadModifier()
  {
    return StaticPolicyT::LargeSegmentPolicy::LOAD_MODIFIER;
  }

  _CCCL_HOST_DEVICE static constexpr BlockLoadAlgorithm LargeSegmentLoadAlgorithm()
  {
    return StaticPolicyT::LargeSegmentPolicy::LOAD_ALGORITHM;
  }

  _CCCL_HOST_DEVICE static constexpr WarpLoadAlgorithm MediumSegmentLoadAlgorithm()
  {
    return StaticPolicyT::MediumSegmentPolicy::LOAD_ALGORITHM;
  }

  _CCCL_HOST_DEVICE static constexpr WarpLoadAlgorithm SmallSegmentLoadAlgorithm()
  {
    return StaticPolicyT::SmallSegmentPolicy::LOAD_ALGORITHM;
  }

  _CCCL_HOST_DEVICE static constexpr WarpStoreAlgorithm MediumSegmentStoreAlgorithm()
  {
    return StaticPolicyT::MediumSegmentPolicy::STORE_ALGORITHM;
  }

  _CCCL_HOST_DEVICE static constexpr WarpStoreAlgorithm SmallSegmentStoreAlgorithm()
  {
    return StaticPolicyT::SmallSegmentPolicy::STORE_ALGORITHM;
  }
};

// TODO(bgruber): remove when we drop the CUB dispatchers in CCCL 4.0
template <typename PolicyT>
_CCCL_HOST_DEVICE SegmentedSortPolicyWrapper<PolicyT> MakeSegmentedSortPolicyWrapper(PolicyT policy)
{
  return SegmentedSortPolicyWrapper<PolicyT>{policy};
}

// TODO(bgruber): remove when we drop the CUB dispatchers in CCCL 4.0
template <typename KeyT, typename ValueT>
struct policy_hub
{
  using DominantT                = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;
  static constexpr int KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 300;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      16,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(7);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(7);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  4 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
  };

  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      19,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(9);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  4 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
  };

  struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(9);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  4 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
  };

  struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 5 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(9);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  4 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
  };

  struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = AgentRadixSortDownsweepPolicy<
      BLOCK_THREADS,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_BITS>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(7);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(KEYS_ONLY ? 11 : 7);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  KEYS_ONLY ? 4 : 8 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_DIRECT,
                                  LOAD_DEFAULT>;
  };

  struct Policy800 : ChainedPolicy<800, Policy800, Policy700>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int PARTITIONING_THRESHOLD = 500;
    using LargeSegmentPolicy                    = AgentRadixSortDownsweepPolicy<
                         BLOCK_THREADS,
                         23,
                         DominantT,
                         BLOCK_LOAD_TRANSPOSE,
                         LOAD_DEFAULT,
                         RADIX_RANK_MEMOIZE,
                         BLOCK_SCAN_WARP_SCANS,
      (sizeof(KeyT) > 1) ? 6 : 4>;

    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(KEYS_ONLY ? 7 : 11);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  KEYS_ONLY ? 4 : 2 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_TRANSPOSE,
                                  LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  32 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_TRANSPOSE,
                                  LOAD_DEFAULT>;
  };

  struct Policy860 : ChainedPolicy<860, Policy860, Policy800>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int PARTITIONING_THRESHOLD = 500;
    using LargeSegmentPolicy                    = AgentRadixSortDownsweepPolicy<
                         BLOCK_THREADS,
                         23,
                         DominantT,
                         BLOCK_LOAD_TRANSPOSE,
                         LOAD_DEFAULT,
                         RADIX_RANK_MEMOIZE,
                         BLOCK_SCAN_WARP_SCANS,
      (sizeof(KeyT) > 1) ? 6 : 4>;

    static constexpr bool LARGE_ITEMS            = sizeof(DominantT) > 4;
    static constexpr int ITEMS_PER_SMALL_THREAD  = Nominal4BItemsToItems<DominantT>(LARGE_ITEMS ? 7 : 9);
    static constexpr int ITEMS_PER_MEDIUM_THREAD = Nominal4BItemsToItems<DominantT>(LARGE_ITEMS ? 9 : 7);

    using SmallSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  LARGE_ITEMS ? 8 : 2 /* Threads per segment */,
                                  ITEMS_PER_SMALL_THREAD,
                                  WARP_LOAD_TRANSPOSE,
                                  LOAD_LDG>;
    using MediumSegmentPolicy =
      AgentSubWarpMergeSortPolicy<BLOCK_THREADS,
                                  16 /* Threads per segment */,
                                  ITEMS_PER_MEDIUM_THREAD,
                                  WARP_LOAD_TRANSPOSE,
                                  LOAD_LDG>;
  };

  using MaxPolicy = Policy860;
};
} // namespace detail::segmented_sort

CUB_NAMESPACE_END
