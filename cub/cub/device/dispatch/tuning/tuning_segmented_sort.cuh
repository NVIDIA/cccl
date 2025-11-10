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
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort
{
template <typename PolicyT, typename = void>
struct SegmentedSortPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE SegmentedSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct SegmentedSortPolicyWrapper<StaticPolicyT,
                                  ::cuda::std::void_t<typename StaticPolicyT::LargeSegmentPolicy,
                                                      typename StaticPolicyT::SmallSegmentPolicy,
                                                      typename StaticPolicyT::MediumSegmentPolicy>> : StaticPolicyT
{
  _CCCL_HOST_DEVICE SegmentedSortPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr auto LargeSegment()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::LargeSegmentPolicy());
  }

  _CCCL_HOST_DEVICE static constexpr auto SmallSegment()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::SmallSegmentPolicy());
  }

  _CCCL_HOST_DEVICE static constexpr auto MediumSegment()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::MediumSegmentPolicy());
  }

  _CCCL_HOST_DEVICE static constexpr int PartitioningThreshold()
  {
    return StaticPolicyT::PARTITIONING_THRESHOLD;
  }

  _CCCL_HOST_DEVICE static constexpr int LargeSegmentRadixBits()
  {
    return StaticPolicyT::LargeSegmentPolicy::RADIX_BITS;
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

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"LargeSegmentPolicy">()    = LargeSegment().EncodedPolicy(),
                  key<"SmallSegmentPolicy">()    = SmallSegment().EncodedPolicy(),
                  key<"MediumSegmentPolicy">()   = MediumSegment().EncodedPolicy(),
                  key<"PartitioningThreshold">() = value<StaticPolicyT::PARTITIONING_THRESHOLD>()>();
  }
#endif
};

template <typename PolicyT>
_CCCL_HOST_DEVICE SegmentedSortPolicyWrapper<PolicyT> MakeSegmentedSortPolicyWrapper(PolicyT policy)
{
  return SegmentedSortPolicyWrapper<PolicyT>{policy};
}

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
