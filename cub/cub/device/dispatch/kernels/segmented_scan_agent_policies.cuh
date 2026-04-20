// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Declare agent policy structs used by kernels implementing algorithms of DeviceSegmentedScan
//! as well as its tunings.
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
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/detail/segmented_scan_helpers.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

template <int BlockThreads,
          int ItemsPerThread,
          BlockLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          BlockStoreAlgorithm StoreAlgorithm,
          BlockScanAlgorithm ScanAlgorithm,
          int MaxSegmentsPerBlock = 1>
struct agent_segmented_scan_policy_t
{
  static_assert(MaxSegmentsPerBlock > 0, "MaxSegmentsPerBlock template value parameter must be positive");

  static constexpr int items_per_thread = ItemsPerThread;
  static constexpr int block_threads    = BlockThreads;

  static constexpr BlockLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier     = LoadModifier;
  static constexpr BlockStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr BlockScanAlgorithm scan_algorithm   = ScanAlgorithm;
  static constexpr int max_segments_per_block          = MaxSegmentsPerBlock;
};

template <typename ComputeT, int NumSegmentsPerBlock>
using agent_block_segmented_scan_compute_t =
  multi_segment_helpers::agent_segmented_scan_compute_t<ComputeT, NumSegmentsPerBlock>;

template <int BlockThreads,
          int ItemsPerThread,
          WarpLoadAlgorithm LoadAlgorithm,
          CacheLoadModifier LoadModifier,
          WarpStoreAlgorithm StoreAlgorithm,
          int MaxSegmentsPerWarp = 1>
struct agent_warp_segmented_scan_policy_t
{
  static_assert(MaxSegmentsPerWarp > 0, "MaxSegmentsPerWarp template value parameter must be positive");

  static constexpr int block_threads    = BlockThreads;
  static constexpr int items_per_thread = ItemsPerThread;

  static constexpr WarpLoadAlgorithm load_algorithm   = LoadAlgorithm;
  static constexpr CacheLoadModifier load_modifier    = LoadModifier;
  static constexpr WarpStoreAlgorithm store_algorithm = StoreAlgorithm;
  static constexpr int max_segments_per_warp          = MaxSegmentsPerWarp;
};

template <typename ComputeT, int NumSegmentsPerWarp>
using agent_warp_segmented_scan_compute_t =
  multi_segment_helpers::agent_segmented_scan_compute_t<ComputeT, NumSegmentsPerWarp>;

template <int BlockThreads, int ItemsPerThread, CacheLoadModifier LoadModifier>
struct agent_thread_segmented_scan_policy_t
{
  static constexpr int block_threads    = BlockThreads;
  static constexpr int items_per_thread = ItemsPerThread;

  static constexpr CacheLoadModifier load_modifier = LoadModifier;
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
