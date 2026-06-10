// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The cub::BlockHistogramAtomicWarpAggregated class provides warp-aggregated atomic methods for
//! constructing block-wide histograms from data samples partitioned across a CUDA thread block.

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

#include <cuda/__cmath/ilog.h>
#include <cuda/__ptx/instructions/bfind.h>
#include <cuda/__ptx/instructions/elect_sync.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! @brief The BlockHistogramAtomicWarpAggregated class provides raw warp intrinsic based methods for constructing
//!        block-wide histograms from data samples partitioned across a CUDA thread block.
template <int Bins, int BlockDimX, int BlockDimY, int BlockDimZ>
struct BlockHistogramAtomicWarpAggregated
{
  static constexpr int warp_threads                     = detail::warp_threads;
  static constexpr int block_threads                    = BlockDimX * BlockDimY * BlockDimZ;
  static constexpr int partial_warp_threads             = block_threads % warp_threads;
  static constexpr int partial_warp_id                  = block_threads / warp_threads;
  static constexpr bool full_warps                      = partial_warp_threads == 0;
  static constexpr int bin_bits                         = (Bins <= 1) ? 1 : ::cuda::ceil_ilog2(Bins);
  static constexpr int ballot_peer_mask_max_bits        = ::cuda::ceil_ilog2(warp_threads);
  static constexpr bool use_ballot_peer_mask            = bin_bits <= ballot_peer_mask_max_bits;
  static constexpr ::cuda::std::uint32_t full_warp_mask = 0xFFFFFFFFu;

  //! Shared memory storage layout type
  struct TempStorage
  {};

  int warp_id;

  //! Constructor
  _CCCL_DEVICE_API _CCCL_FORCEINLINE BlockHistogramAtomicWarpAggregated(TempStorage&)
      : warp_id(cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ) / warp_threads)
  {}

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::uint32_t NativePeerMaskImpl(unsigned bin) const
  {
    const auto active_mask = (full_warps || warp_id != partial_warp_id) ? full_warp_mask : ::__activemask();
    return ::__match_any_sync(active_mask, bin);
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::uint32_t NativePeerMask(unsigned bin) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70, (return NativePeerMaskImpl(bin);), (return BallotPeerMask(bin);));
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::uint32_t BallotPeerMask(unsigned bin) const
  {
    constexpr auto partial_warp_mask = (::cuda::std::uint32_t{1} << partial_warp_threads) - 1;
    const auto warp_mask             = (full_warps || warp_id != partial_warp_id) ? full_warp_mask : partial_warp_mask;
    return cub::MatchAny<bin_bits>(bin) & warp_mask;
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::uint32_t PeerMask(unsigned bin) const
  {
    if constexpr (use_ballot_peer_mask)
    {
      return BallotPeerMask(bin);
    }
    else
    {
      return NativePeerMask(bin);
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE bool IsLeader(::cuda::std::uint32_t peer_mask) const
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      (return ::cuda::ptx::elect_sync(peer_mask);),
                      (return ::cuda::ptx::get_sreg_laneid() == ::cuda::ptx::bfind(peer_mask);));
  }

  template <typename CounterT>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Add(CounterT& counter_ref, CounterT count) const
  {
    atomicAdd(&counter_ref, count);
  }

  //! @brief Composite data onto an existing histogram
  //!
  //! @param[in] items
  //!   Calling thread's input values to histogram
  //!
  //! @param[out] histogram
  //!   Reference to shared or global memory histogram
  //!
  //! @tparam CounterT
  //!   Counter type accepted by CUDA atomicAdd
  template <typename T, typename CounterT, int ItemsPerThread>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Composite(const T (&items)[ItemsPerThread], CounterT histogram[Bins]) const
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      const auto bin_value = static_cast<::cuda::std::uint64_t>(items[i]);
      _CCCL_ASSERT(bin_value < static_cast<::cuda::std::uint64_t>(Bins), "sample value must be in [0, Bins)");

      const auto bin       = static_cast<unsigned>(bin_value);
      const auto peer_mask = this->PeerMask(bin);

      if (IsLeader(peer_mask))
      {
        const auto count = static_cast<CounterT>(::cuda::std::popcount(peer_mask));
        Add(histogram[bin], count);
      }
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
