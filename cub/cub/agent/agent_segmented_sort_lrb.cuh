// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{

//! Shape-grid policy for 2-D LRB. `MaxCooperativeThreads` is the largest power-of-two
//! cooperative group that still fits one CTA (1024 for 4-byte keys, 512 for 8-byte).
template <int MaxCooperativeThreads = 1024,
          int IptMin                = 9,
          int IptMax                = 15,
          int BlockThreads          = 256,
          int SmallCtaThreads       = 256>
struct lrb_policy
{
  static constexpr int max_cooperative_threads = MaxCooperativeThreads;
  static constexpr int ipt_min                 = IptMin;
  static constexpr int ipt_max                 = IptMax;
  static constexpr int block_threads           = BlockThreads;
  static constexpr int small_cta_threads       = SmallCtaThreads;
  static constexpr int large_cta_threads       = MaxCooperativeThreads;
  static constexpr int warp_threads            = 32;

  static constexpr int ipt_choices = IptMax - IptMin + 1;

  static constexpr int num_warp_thread_choices        = 6; // 1, 2, 4, 8, 16, 32
  static constexpr int num_small_block_thread_choices = 3; // 64, 128, 256
  static constexpr int num_large_block_thread_choices = (MaxCooperativeThreads >= 1024) ? 2 : 1;

  static_assert(MaxCooperativeThreads == 512 || MaxCooperativeThreads == 1024,
                "LRB planner currently supports MaxCooperativeThreads in {512, 1024}");

  static constexpr int warp_bin_count        = num_warp_thread_choices * ipt_choices;
  static constexpr int small_block_bin_count = num_small_block_thread_choices * ipt_choices;
  static constexpr int large_block_bin_count = num_large_block_thread_choices * ipt_choices;
  static constexpr int overflow_bin_count    = 1;

  static constexpr int total_bin_count =
    warp_bin_count + small_block_bin_count + large_block_bin_count + overflow_bin_count;
};

enum class lrb_tier : int
{
  none         = -1,
  warp         = 0,
  small_block  = 1,
  large_block  = 2,
  overflow     = 3
};

struct lrb_shape
{
  lrb_tier tier{lrb_tier::none};
  int threads{0};
  int items_per_thread{0};
  int bin{0};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool assigned() const
  {
    return tier != lrb_tier::none;
  }
};

template <typename Policy>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int warp_threads_for_bin(int bin)
{
  return 1 << (bin / Policy::ipt_choices);
}

template <typename Policy>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int small_block_threads_for_bin(int bin)
{
  return 64 << (bin / Policy::ipt_choices);
}

template <typename Policy>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE int large_block_threads_for_bin(int bin)
{
  return 512 << (bin / Policy::ipt_choices);
}

//! Closed-form 2-D LRB map used by count and scatter. Segments with length <= 1 are not binned
//! (CUB already treats those as empty: `end - 1 <= begin`).
template <typename Policy, typename OffsetT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE lrb_shape map_segment_length(OffsetT length)
{
  lrb_shape shape{};
  if (length <= OffsetT{1})
  {
    return shape;
  }

  using unsigned_t = ::cuda::std::make_unsigned_t<OffsetT>;
  const unsigned_t n       = static_cast<unsigned_t>(length);
  const unsigned_t min_t   = ::cuda::ceil_div(n, static_cast<unsigned_t>(Policy::ipt_max));
  const unsigned_t threads = ::cuda::std::bit_ceil(min_t);
  const int ipt            = static_cast<int>(
    (::cuda::std::max) (::cuda::ceil_div(n, threads), static_cast<unsigned_t>(Policy::ipt_min)));
  const int ipt_slot = ipt - Policy::ipt_min;

  shape.threads           = static_cast<int>(threads);
  shape.items_per_thread  = ipt;

  if (threads <= 32)
  {
    shape.tier = lrb_tier::warp;
    shape.bin  = static_cast<int>(::cuda::std::countr_zero(threads)) * Policy::ipt_choices + ipt_slot;
  }
  else if (threads <= 256)
  {
    shape.tier = lrb_tier::small_block;
    shape.bin  = (static_cast<int>(::cuda::std::countr_zero(threads)) - Policy::num_warp_thread_choices)
              * Policy::ipt_choices
            + ipt_slot;
  }
  else if (threads <= static_cast<unsigned_t>(Policy::max_cooperative_threads))
  {
    shape.tier = lrb_tier::large_block;
    shape.bin =
      (static_cast<int>(::cuda::std::countr_zero(threads)) - 9) * Policy::ipt_choices + ipt_slot;
  }
  else
  {
    shape.tier = lrb_tier::overflow;
    shape.bin  = 0;
  }
  return shape;
}

template <typename CountT>
struct lrb_summary
{
  CountT num_warp{};
  CountT num_small_block{};
  CountT num_large_block{};
  CountT num_overflow{};
  CountT num_assigned{};
};

template <typename CountT>
struct lrb_plan
{
  CountT* d_segment_ids{};
  CountT* d_warp_bin_offsets{};
  CountT* d_small_block_bin_offsets{};
  CountT* d_large_block_bin_offsets{};
  CountT* d_overflow_offsets{};
  CountT* d_warp_group_offsets{};
  CountT* d_small_block_group_offsets{};
  CountT* d_large_block_group_offsets{};
  lrb_summary<CountT>* d_summary{};
};

template <typename Policy, typename CountT>
struct lrb_histograms
{
  CountT* warp{};
  CountT* small_block{};
  CountT* large_block{};
  CountT* overflow{};
};

template <typename Policy, typename CountT>
_CCCL_DEVICE _CCCL_FORCEINLINE void zero_smem_hist(lrb_histograms<Policy, CountT> hist)
{
  for (int i = threadIdx.x; i < Policy::warp_bin_count; i += blockDim.x)
  {
    hist.warp[i] = CountT{};
  }
  for (int i = threadIdx.x; i < Policy::small_block_bin_count; i += blockDim.x)
  {
    hist.small_block[i] = CountT{};
  }
  for (int i = threadIdx.x; i < Policy::large_block_bin_count; i += blockDim.x)
  {
    hist.large_block[i] = CountT{};
  }
  if (threadIdx.x == 0)
  {
    hist.overflow[0] = CountT{};
  }
}

template <typename Policy, typename CountT>
_CCCL_DEVICE _CCCL_FORCEINLINE void
atomic_add_nonzero(CountT* smem, CountT* global, int n)
{
  for (int i = threadIdx.x; i < n; i += blockDim.x)
  {
    if (smem[i] > CountT{})
    {
      atomicAdd(global + i, smem[i]);
    }
  }
}

template <typename Policy, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename CountT>
struct AgentSegmentedSortLrbCount
{
  struct TempStorage
  {
    CountT warp[Policy::warp_bin_count];
    CountT small_block[Policy::small_block_bin_count];
    CountT large_block[Policy::large_block_bin_count];
    CountT overflow[Policy::overflow_bin_count];
  };

  TempStorage& storage;
  BeginOffsetIteratorT d_begin_offsets;
  EndOffsetIteratorT d_end_offsets;
  int num_segments;
  lrb_histograms<Policy, CountT> global_hist;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSegmentedSortLrbCount(
    TempStorage& storage,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    lrb_histograms<Policy, CountT> global_hist)
      : storage(storage)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , num_segments(num_segments)
      , global_hist(global_hist)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    lrb_histograms<Policy, CountT> local{
      storage.warp, storage.small_block, storage.large_block, storage.overflow};
    zero_smem_hist<Policy>(local);
    __syncthreads();

    const int segment_id = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    if (segment_id < num_segments)
    {
      const auto begin  = d_begin_offsets[segment_id];
      const auto end    = d_end_offsets[segment_id];
      const auto length = (end > begin) ? (end - begin) : decltype(end - begin){};
      const lrb_shape shape = map_segment_length<Policy>(length);
      if (shape.tier == lrb_tier::warp)
      {
        atomicAdd(local.warp + shape.bin, CountT{1});
      }
      else if (shape.tier == lrb_tier::small_block)
      {
        atomicAdd(local.small_block + shape.bin, CountT{1});
      }
      else if (shape.tier == lrb_tier::large_block)
      {
        atomicAdd(local.large_block + shape.bin, CountT{1});
      }
      else if (shape.tier == lrb_tier::overflow)
      {
        atomicAdd(local.overflow, CountT{1});
      }
    }

    __syncthreads();
    atomic_add_nonzero<Policy>(local.warp, global_hist.warp, Policy::warp_bin_count);
    atomic_add_nonzero<Policy>(local.small_block, global_hist.small_block, Policy::small_block_bin_count);
    atomic_add_nonzero<Policy>(local.large_block, global_hist.large_block, Policy::large_block_bin_count);
    if (threadIdx.x == 0 && local.overflow[0] > CountT{})
    {
      atomicAdd(global_hist.overflow, local.overflow[0]);
    }
  }
};

template <typename CountT>
_CCCL_DEVICE _CCCL_FORCEINLINE void exclusive_scan_counts(const CountT* counts, CountT* offsets, int n, CountT base)
{
  CountT acc = base;
  offsets[0] = acc;
  for (int i = 0; i < n; ++i)
  {
    acc += counts[i];
    offsets[i + 1] = acc;
  }
}

template <typename CountT>
_CCCL_DEVICE _CCCL_FORCEINLINE void copy_offsets(const CountT* src, CountT* dst, int n_plus_one)
{
  for (int i = 0; i < n_plus_one; ++i)
  {
    dst[i] = src[i];
  }
}

//! Single-thread (or single-lane) metadata: exclusive ID ranges and logical launch-group offsets.
template <typename Policy, typename CountT>
_CCCL_DEVICE _CCCL_FORCEINLINE void finalize_lrb_metadata(
  const lrb_histograms<Policy, CountT> counts,
  CountT* warp_offsets,
  CountT* small_offsets,
  CountT* large_offsets,
  CountT* overflow_offsets,
  CountT* warp_offsets_atomic,
  CountT* small_offsets_atomic,
  CountT* large_offsets_atomic,
  CountT* overflow_offsets_atomic,
  CountT* warp_group_offsets,
  CountT* small_group_offsets,
  CountT* large_group_offsets,
  lrb_summary<CountT>* summary)
{
  exclusive_scan_counts(counts.warp, warp_offsets, Policy::warp_bin_count, CountT{0});
  exclusive_scan_counts(
    counts.small_block, small_offsets, Policy::small_block_bin_count, warp_offsets[Policy::warp_bin_count]);
  exclusive_scan_counts(
    counts.large_block, large_offsets, Policy::large_block_bin_count, small_offsets[Policy::small_block_bin_count]);
  overflow_offsets[0] = large_offsets[Policy::large_block_bin_count];
  overflow_offsets[1] = overflow_offsets[0] + counts.overflow[0];

  copy_offsets(warp_offsets, warp_offsets_atomic, Policy::warp_bin_count + 1);
  copy_offsets(small_offsets, small_offsets_atomic, Policy::small_block_bin_count + 1);
  copy_offsets(large_offsets, large_offsets_atomic, Policy::large_block_bin_count + 1);
  overflow_offsets_atomic[0] = overflow_offsets[0];
  overflow_offsets_atomic[1] = overflow_offsets[1];

  warp_group_offsets[0] = CountT{0};
  for (int i = 0; i < Policy::warp_bin_count; ++i)
  {
    const int threads        = warp_threads_for_bin<Policy>(i);
    const int segs_per_group = Policy::warp_threads / threads;
    warp_group_offsets[i + 1] =
      warp_group_offsets[i] + static_cast<CountT>(::cuda::ceil_div(counts.warp[i], static_cast<CountT>(segs_per_group)));
  }

  small_group_offsets[0] = CountT{0};
  for (int i = 0; i < Policy::small_block_bin_count; ++i)
  {
    const int threads      = small_block_threads_for_bin<Policy>(i);
    const int segs_per_cta = Policy::small_cta_threads / threads;
    small_group_offsets[i + 1] =
      small_group_offsets[i] + static_cast<CountT>(::cuda::ceil_div(counts.small_block[i], static_cast<CountT>(segs_per_cta)));
  }

  large_group_offsets[0] = CountT{0};
  for (int i = 0; i < Policy::large_block_bin_count; ++i)
  {
    const int threads      = large_block_threads_for_bin<Policy>(i);
    const int segs_per_cta = Policy::large_cta_threads / threads;
    if (segs_per_cta < 1)
    {
      large_group_offsets[i + 1] = large_group_offsets[i];
      continue;
    }
    large_group_offsets[i + 1] =
      large_group_offsets[i] + static_cast<CountT>(::cuda::ceil_div(counts.large_block[i], static_cast<CountT>(segs_per_cta)));
  }

  summary->num_warp         = warp_offsets[Policy::warp_bin_count];
  summary->num_small_block  = small_offsets[Policy::small_block_bin_count] - summary->num_warp;
  summary->num_large_block  = large_offsets[Policy::large_block_bin_count] - small_offsets[Policy::small_block_bin_count];
  summary->num_overflow     = counts.overflow[0];
  summary->num_assigned     = overflow_offsets[1];
}

template <typename Policy, typename BeginOffsetIteratorT, typename EndOffsetIteratorT, typename CountT>
struct AgentSegmentedSortLrbScatter
{
  struct TempStorage
  {
    CountT warp[Policy::warp_bin_count];
    CountT small_block[Policy::small_block_bin_count];
    CountT large_block[Policy::large_block_bin_count];
    CountT overflow[Policy::overflow_bin_count];
    CountT warp_pos[Policy::warp_bin_count];
    CountT small_pos[Policy::small_block_bin_count];
    CountT large_pos[Policy::large_block_bin_count];
    CountT overflow_pos[Policy::overflow_bin_count];
  };

  TempStorage& storage;
  BeginOffsetIteratorT d_begin_offsets;
  EndOffsetIteratorT d_end_offsets;
  int num_segments;
  CountT* warp_offsets_atomic;
  CountT* small_offsets_atomic;
  CountT* large_offsets_atomic;
  CountT* overflow_offsets_atomic;
  CountT* d_segment_ids;

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentSegmentedSortLrbScatter(
    TempStorage& storage,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    int num_segments,
    CountT* warp_offsets_atomic,
    CountT* small_offsets_atomic,
    CountT* large_offsets_atomic,
    CountT* overflow_offsets_atomic,
    CountT* d_segment_ids)
      : storage(storage)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , num_segments(num_segments)
      , warp_offsets_atomic(warp_offsets_atomic)
      , small_offsets_atomic(small_offsets_atomic)
      , large_offsets_atomic(large_offsets_atomic)
      , overflow_offsets_atomic(overflow_offsets_atomic)
      , d_segment_ids(d_segment_ids)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void Process()
  {
    lrb_histograms<Policy, CountT> local{
      storage.warp, storage.small_block, storage.large_block, storage.overflow};
    zero_smem_hist<Policy>(local);
    __syncthreads();

    int segment_id          = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    lrb_shape shape{};
    CountT local_rank       = CountT{};
    const bool in_range     = segment_id < num_segments;
    if (!in_range)
    {
      segment_id = -1;
    }
    else
    {
      const auto begin  = d_begin_offsets[segment_id];
      const auto end    = d_end_offsets[segment_id];
      const auto length = (end > begin) ? (end - begin) : decltype(end - begin){};
      shape             = map_segment_length<Policy>(length);
      if (shape.tier == lrb_tier::warp)
      {
        local_rank = atomicAdd(local.warp + shape.bin, CountT{1});
      }
      else if (shape.tier == lrb_tier::small_block)
      {
        local_rank = atomicAdd(local.small_block + shape.bin, CountT{1});
      }
      else if (shape.tier == lrb_tier::large_block)
      {
        local_rank = atomicAdd(local.large_block + shape.bin, CountT{1});
      }
      else if (shape.tier == lrb_tier::overflow)
      {
        local_rank = atomicAdd(local.overflow, CountT{1});
      }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < Policy::warp_bin_count; i += blockDim.x)
    {
      if (local.warp[i] > CountT{})
      {
        storage.warp_pos[i] = atomicAdd(warp_offsets_atomic + i, local.warp[i]);
      }
    }
    for (int i = threadIdx.x; i < Policy::small_block_bin_count; i += blockDim.x)
    {
      if (local.small_block[i] > CountT{})
      {
        storage.small_pos[i] = atomicAdd(small_offsets_atomic + i, local.small_block[i]);
      }
    }
    for (int i = threadIdx.x; i < Policy::large_block_bin_count; i += blockDim.x)
    {
      if (local.large_block[i] > CountT{})
      {
        storage.large_pos[i] = atomicAdd(large_offsets_atomic + i, local.large_block[i]);
      }
    }
    if (threadIdx.x == 0 && local.overflow[0] > CountT{})
    {
      storage.overflow_pos[0] = atomicAdd(overflow_offsets_atomic, local.overflow[0]);
    }
    __syncthreads();

    if (segment_id >= 0 && shape.assigned())
    {
      CountT write_pos{};
      if (shape.tier == lrb_tier::warp)
      {
        write_pos = storage.warp_pos[shape.bin] + local_rank;
      }
      else if (shape.tier == lrb_tier::small_block)
      {
        write_pos = storage.small_pos[shape.bin] + local_rank;
      }
      else if (shape.tier == lrb_tier::large_block)
      {
        write_pos = storage.large_pos[shape.bin] + local_rank;
      }
      else
      {
        write_pos = storage.overflow_pos[0] + local_rank;
      }
      d_segment_ids[write_pos] = static_cast<CountT>(segment_id);
    }
  }
};

} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END
