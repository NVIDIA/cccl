// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * The cub::BlockHistogramAtomicWarpAggregated class provides warp-aggregated atomic methods for
 * constructing block-wide histograms from data samples partitioned across a CUDA thread block.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/get_sreg.h>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
namespace cg = cooperative_groups;

/**
 * @brief The BlockHistogramAtomicWarpAggregated class provides raw warp intrinsic based methods for constructing
 *        block-wide histograms from data samples partitioned across a CUDA thread block.
 */
template <int Bins>
struct BlockHistogramAtomicWarpAggregated
{
  /// Shared memory storage layout type
  struct TempStorage
  {};

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramAtomicWarpAggregated(TempStorage& temp_storage) {}

  /**
   * @brief Composite data onto an existing histogram
   *
   * @param[in] items
   *   Calling thread's input values to histogram
   *
   * @param[out] histogram
   *   Reference to shared/device-accessible memory histogram
   */
  template <typename T, typename CounterT, int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT histogram[Bins])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      const unsigned int bin       = static_cast<unsigned int>(items[i]);
      const unsigned int peer_mask = __match_any_sync(__activemask(), bin);
      const int leader             = __ffs(static_cast<int>(peer_mask)) - 1;

      if (static_cast<int>(::cuda::ptx::get_sreg_laneid()) == leader)
      {
        atomicAdd(histogram + bin, static_cast<CounterT>(__popc(peer_mask)));
      }
    }
  }
};

/**
 * @brief The BlockHistogramAtomicWarpAggregatedCg class provides cooperative-groups based methods for constructing
 *        block-wide histograms from data samples partitioned across a CUDA thread block.
 */
template <int Bins>
struct BlockHistogramAtomicWarpAggregatedCg
{
  /// Shared memory storage layout type
  struct TempStorage
  {};

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramAtomicWarpAggregatedCg(TempStorage& temp_storage) {}

  /**
   * @brief Composite data onto an existing histogram
   *
   * @param[in] items
   *   Calling thread's input values to histogram
   *
   * @param[out] histogram
   *   Reference to shared/device-accessible memory histogram
   */
  template <typename T, typename CounterT, int ITEMS_PER_THREAD>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT histogram[Bins])
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      cg::coalesced_group active = cg::coalesced_threads();
      const T bin                = items[i];
      auto bin_group             = cg::labeled_partition(active, bin);
      const CounterT votes       = cg::reduce(bin_group, CounterT{1}, cg::plus<CounterT>());

      if (bin_group.thread_rank() == 0)
      {
        atomicAdd(histogram + bin, votes);
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
