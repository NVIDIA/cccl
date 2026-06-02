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

#include <cub/util_ptx.cuh>

#include <cuda/__ptx/instructions/get_sreg.h>

CUB_NAMESPACE_BEGIN
namespace detail
{
/**
 * @brief The BlockHistogramAtomicWarpAggregated class provides raw warp intrinsic based methods for constructing
 *        block-wide histograms from data samples partitioned across a CUDA thread block.
 */
template <int Bins, int BlockDimX, int BlockDimY, int BlockDimZ>
struct BlockHistogramAtomicWarpAggregated
{
  static constexpr int BLOCK_THREADS        = BlockDimX * BlockDimY * BlockDimZ;
  static constexpr int PARTIAL_WARP_THREADS = BLOCK_THREADS % detail::warp_threads;
  static constexpr int PARTIAL_WARP_ID      = BLOCK_THREADS / detail::warp_threads;

  /// Shared memory storage layout type
  struct TempStorage
  {};

  unsigned int linear_tid;

  /// Constructor
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramAtomicWarpAggregated(TempStorage& temp_storage)
      : linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  template <int BIN_BITS>
  _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int FallbackPeerMask(unsigned int bin) const
  {
    if constexpr (PARTIAL_WARP_THREADS == 0)
    {
      return MatchAny<BIN_BITS>(bin);
    }
    else
    {
      const unsigned int warp_id = linear_tid / detail::warp_threads;
      return (warp_id == PARTIAL_WARP_ID) ? MatchAny<BIN_BITS, PARTIAL_WARP_THREADS>(bin) : MatchAny<BIN_BITS>(bin);
    }
  }

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
    constexpr int BIN_BITS = (Bins <= 1) ? 1 : Log2<Bins>::VALUE;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      const unsigned int bin = static_cast<unsigned int>(items[i]);
      unsigned int peer_mask;
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
                        (peer_mask = __match_any_sync(__activemask(), bin);),
                        (peer_mask = FallbackPeerMask<BIN_BITS>(bin);));
      const int leader = __ffs(static_cast<int>(peer_mask)) - 1;

      if (static_cast<int>(::cuda::ptx::get_sreg_laneid()) == leader)
      {
        atomicAdd(histogram + bin, static_cast<CounterT>(__popc(peer_mask)));
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
