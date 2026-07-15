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

//! Policy for the large-segment radix sort step in @ref DeviceSegmentedSort.
struct SegmentedSortRadixSortPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  BlockLoadAlgorithm load_algorithm; //!< The @ref BlockLoadAlgorithm used for loading keys from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading keys from global memory
  RadixRankAlgorithm rank_algorithm; //!< The @ref RadixRankAlgorithm used for ranking keys within a block
  BlockScanAlgorithm scan_algorithm; //!< The @ref BlockScanAlgorithm used for the internal digit-count scan
  int radix_bits; //!< Number of bits per radix digit pass

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const SegmentedSortRadixSortPolicy& lhs, const SegmentedSortRadixSortPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.rank_algorithm == rhs.rank_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const SegmentedSortRadixSortPolicy& lhs, const SegmentedSortRadixSortPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const SegmentedSortRadixSortPolicy& p)
  {
    return os
        << "SegmentedSortRadixSortPolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .rank_algorithm = " << p.rank_algorithm
        << ", .scan_algorithm = " << p.scan_algorithm << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // _CCCL_HOSTED()
};

//! Policy for the sub-warp merge sort step in @ref DeviceSegmentedSort (small/medium segments).
struct SegmentedSortSubWarpMergeSortPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int threads_per_warp; //!< Number of threads assigned to sort a single segment
  int items_per_thread; //!< Number of items processed per thread
  WarpLoadAlgorithm load_algorithm; //!< The @ref WarpLoadAlgorithm used for loading items from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  WarpStoreAlgorithm store_algorithm; //!< The @ref WarpStoreAlgorithm used for storing items to global memory

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int segments_per_block() const noexcept
  {
    return threads_per_block / threads_per_warp;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int items_per_tile() const noexcept
  {
    return threads_per_warp * items_per_thread;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const SegmentedSortSubWarpMergeSortPolicy& lhs, const SegmentedSortSubWarpMergeSortPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.threads_per_warp == rhs.threads_per_warp
        && lhs.items_per_thread == rhs.items_per_thread && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.store_algorithm == rhs.store_algorithm;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const SegmentedSortSubWarpMergeSortPolicy& lhs, const SegmentedSortSubWarpMergeSortPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const SegmentedSortSubWarpMergeSortPolicy& p)
  {
    return os
        << "SegmentedSortSubWarpMergeSortPolicy { .threads_per_block = " << p.threads_per_block
        << ", .threads_per_warp = " << p.threads_per_warp << ", .items_per_thread = " << p.items_per_thread
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

//! Top-level tuning policy for @ref DeviceSegmentedSort.
struct SegmentedSortPolicy
{
  SegmentedSortRadixSortPolicy large_segment; //!< Policy used for segments sorted via radix sort
  SegmentedSortSubWarpMergeSortPolicy medium_segment; //!< Policy used for medium-sized segments
  SegmentedSortSubWarpMergeSortPolicy small_segment; //!< Policy used for the smallest segments
  int partitioning_threshold; //!< Number of segments above which different algorithms will be used for different size
                              //!< buckets

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const SegmentedSortPolicy& lhs, const SegmentedSortPolicy& rhs) noexcept
  {
    return lhs.large_segment == rhs.large_segment && lhs.medium_segment == rhs.medium_segment
        && lhs.small_segment == rhs.small_segment && lhs.partitioning_threshold == rhs.partitioning_threshold;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const SegmentedSortPolicy& lhs, const SegmentedSortPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const SegmentedSortPolicy& p)
  {
    return os << "SegmentedSortPolicy { .large_segment = " << p.large_segment
              << ", .medium_segment = " << p.medium_segment << ", .small_segment = " << p.small_segment
              << ", .partitioning_threshold = " << p.partitioning_threshold << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::segmented_sort
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_sort_policy_selector = policy_selector<T, SegmentedSortPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;
  int value_size;
  bool keys_only;

  _CCCL_HOST_DEVICE_API constexpr auto __dominant_size() const
  {
    return ::cuda::std::max(key_size, value_size);
  }

  _CCCL_HOST_DEVICE_API constexpr auto __make_scaled_segmented_radix_sort_policy(
    int nominal_4B_threads_per_block,
    int nominal_4B_items_per_thread,
    BlockLoadAlgorithm load_algorithm,
    CacheLoadModifier load_modifier,
    RadixRankAlgorithm rank_algorithm,
    BlockScanAlgorithm scan_algorithm,
    int radix_bits) const
  {
    const auto scaled = scale_reg_bound(nominal_4B_threads_per_block, nominal_4B_items_per_thread, __dominant_size());
    return SegmentedSortRadixSortPolicy{
      scaled.threads_per_block,
      scaled.items_per_thread,
      load_algorithm,
      load_modifier,
      rank_algorithm,
      scan_algorithm,
      radix_bits};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> SegmentedSortPolicy
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
      return SegmentedSortPolicy{
        __make_scaled_segmented_radix_sort_policy(
          256, 23, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        SegmentedSortSubWarpMergeSortPolicy{256, 16, medium_itp, WARP_LOAD_TRANSPOSE, LOAD_LDG, WARP_STORE_DIRECT},
        SegmentedSortSubWarpMergeSortPolicy{
          256, large_items ? 8 : 2, small_itp, WARP_LOAD_TRANSPOSE, LOAD_LDG, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      const int radix_bits = key_size > 1 ? 6 : 4;
      const int small_itp  = scale_items(9);
      const int medium_itp = scale_items(keys_only ? 7 : 11);
      return SegmentedSortPolicy{
        __make_scaled_segmented_radix_sort_policy(
          256, 23, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        SegmentedSortSubWarpMergeSortPolicy{256, 32, medium_itp, WARP_LOAD_TRANSPOSE, LOAD_DEFAULT, WARP_STORE_DIRECT},
        SegmentedSortSubWarpMergeSortPolicy{
          256, keys_only ? 4 : 2, small_itp, WARP_LOAD_TRANSPOSE, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{7, 0})
    {
      const int radix_bits = key_size > 1 ? 6 : 4;
      const int small_itp  = scale_items(7);
      const int medium_itp = scale_items(keys_only ? 11 : 7);
      return SegmentedSortPolicy{
        __make_scaled_segmented_radix_sort_policy(
          256, 19, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        SegmentedSortSubWarpMergeSortPolicy{256, 32, medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        SegmentedSortSubWarpMergeSortPolicy{
          256, keys_only ? 4 : 8, small_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{6, 2})
    {
      const int radix_bits       = key_size > 1 ? 5 : 4;
      const int small_medium_itp = scale_items(9);
      return SegmentedSortPolicy{
        __make_scaled_segmented_radix_sort_policy(
          256, 16, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, radix_bits),
        SegmentedSortSubWarpMergeSortPolicy{
          256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        SegmentedSortSubWarpMergeSortPolicy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{6, 1})
    {
      const int radix_bits       = key_size > 1 ? 6 : 4;
      const int small_medium_itp = scale_items(9);
      return SegmentedSortPolicy{
        __make_scaled_segmented_radix_sort_policy(
          256, 19, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, radix_bits),
        SegmentedSortSubWarpMergeSortPolicy{
          256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        SegmentedSortSubWarpMergeSortPolicy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      const int radix_bits       = key_size > 1 ? 6 : 4;
      const int small_medium_itp = scale_items(9);
      return SegmentedSortPolicy{
        __make_scaled_segmented_radix_sort_policy(
          256, 19, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, radix_bits),
        SegmentedSortSubWarpMergeSortPolicy{
          256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        SegmentedSortSubWarpMergeSortPolicy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
        500};
    }

    // default for SM50
    const int radix_bits       = key_size > 1 ? 6 : 4;
    const int small_medium_itp = scale_items(7);
    return SegmentedSortPolicy{
      __make_scaled_segmented_radix_sort_policy(
        256, 16, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE, BLOCK_SCAN_RAKING_MEMOIZE, radix_bits),
      SegmentedSortSubWarpMergeSortPolicy{256, 32, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
      SegmentedSortSubWarpMergeSortPolicy{256, 4, small_medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT, WARP_STORE_DIRECT},
      300};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_sort_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyT, typename ValueT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> SegmentedSortPolicy
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

  _CCCL_HOST_DEVICE static constexpr int LargeSegmentThreadsPerBlock()
  {
    return StaticPolicyT::LargeSegmentPolicy::BLOCK_THREADS;
  }

  _CCCL_HOST_DEVICE static constexpr int LargeSegmentItemsPerThread()
  {
    return StaticPolicyT::LargeSegmentPolicy::ITEMS_PER_THREAD;
  }

  _CCCL_HOST_DEVICE static constexpr int SmallSegmentThreadsPerBlock()
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

  struct Policy500 : detail::chained_policy<500, Policy500, Policy500>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 300;

    using LargeSegmentPolicy = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       4 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       32 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
  };

  struct Policy600 : detail::chained_policy<600, Policy600, Policy500>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       4 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       32 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
  };

  struct Policy610 : detail::chained_policy<610, Policy610, Policy600>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       4 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       32 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
  };

  struct Policy620 : detail::chained_policy<620, Policy620, Policy610>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 5 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       4 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       32 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
  };

  struct Policy700 : detail::chained_policy<700, Policy700, Policy620>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int RADIX_BITS             = sizeof(KeyT) > 1 ? 6 : 4;
    static constexpr int PARTITIONING_THRESHOLD = 500;

    using LargeSegmentPolicy = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       KEYS_ONLY ? 4 : 8 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       32 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_DIRECT,
                                       LOAD_DEFAULT>;
  };

  struct Policy800 : detail::chained_policy<800, Policy800, Policy700>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int PARTITIONING_THRESHOLD = 500;
    using LargeSegmentPolicy                    = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       KEYS_ONLY ? 4 : 2 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_TRANSPOSE,
                                       LOAD_DEFAULT>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       32 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_TRANSPOSE,
                                       LOAD_DEFAULT>;
  };

  struct Policy860 : detail::chained_policy<860, Policy860, Policy800>
  {
    static constexpr int BLOCK_THREADS          = 256;
    static constexpr int PARTITIONING_THRESHOLD = 500;
    using LargeSegmentPolicy                    = detail::agent_radix_sort_downsweep_policy<
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
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       LARGE_ITEMS ? 8 : 2 /* Threads per segment */,
                                       ITEMS_PER_SMALL_THREAD,
                                       WARP_LOAD_TRANSPOSE,
                                       LOAD_LDG>;
    using MediumSegmentPolicy =
      agent_sub_warp_merge_sort_policy<BLOCK_THREADS,
                                       16 /* Threads per segment */,
                                       ITEMS_PER_MEDIUM_THREAD,
                                       WARP_LOAD_TRANSPOSE,
                                       LOAD_LDG>;
  };

  using MaxPolicy = Policy860;
};
} // namespace detail::segmented_sort

CUB_NAMESPACE_END
