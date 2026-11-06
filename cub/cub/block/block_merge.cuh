// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/block/block_merge_sort.cuh>
#include <cub/detail/log2.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_ptx.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>

CUB_NAMESPACE_BEGIN

//! @brief Shared-memory strategies for BlockMerge.
enum BlockMergeAlgorithm
{
  //! Merge between two shared buffers, with one block barrier per round.
  BLOCK_MERGE_PING_PONG,
  //! Stage each round in registers, using one shared buffer and two barriers per intermediate round.
  BLOCK_MERGE_REGISTER_STAGED
};

namespace detail
{
// Bracket the merge-path crossing near a proportional estimate, then reuse CUB's binary search.
template <typename KeyT, typename CompareOp>
[[nodiscard]] _CCCL_DEVICE_API int
block_merge_path(const KeyT* keys1, const KeyT* keys2, int count1, int count2, int diagonal, CompareOp compare_op)
{
  int first = (::cuda::std::max) (0, diagonal - count2);
  int last  = (::cuda::std::min) (diagonal, count1);
  if (first < last)
  {
    const int estimate = static_cast<int>(
      __fdividef(static_cast<float>(diagonal) * static_cast<float>(count1), static_cast<float>(count1 + count2)));
    const int pivot = (::cuda::std::min) ((::cuda::std::max) (estimate, first), last - 1);
    const bool left = compare_op(keys2[diagonal - 1 - pivot], keys1[pivot]);
    if (left)
    {
      last = pivot;
    }
    else
    {
      first = pivot + 1;
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int probe = 0, distance = 16; probe < 3; ++probe, distance *= 4)
    {
      const int index = left ? pivot - distance : pivot + distance;
      if (index < first || index >= last)
      {
        break;
      }
      if (compare_op(keys2[diagonal - 1 - index], keys1[index]))
      {
        last = index;
        if (!left)
        {
          break;
        }
      }
      else
      {
        first = index + 1;
        if (left)
        {
          break;
        }
      }
    }
  }
  const int span = last - first;
  return first + MergePath(keys1 + first, keys2 + diagonal - last, span, span, span, compare_op);
}
} // namespace detail

//! @brief Cooperatively merges up to MaxRuns sorted runs already in shared memory.
//!
//! Runs occupy consecutive ranges [offsets[r], offsets[r + 1]) of keys, and empty runs are allowed.
//! All threads must participate with identical pointers, offsets, run count, and comparator.
//! The result is stable: equal keys from an earlier run precede equal keys from a later run.
//!
//! @rst
//! The caller initializes the live prefix and one readable slack key, then synchronizes before calling.
//! The input buffer has ``BlockDimX * BlockDimY * BlockDimZ * ItemsPerThread + 1`` key slots.
//! ``offsets`` has ``num_runs + 1`` nondecreasing entries, starts at zero, and ends at the live key count.
//! The live count must fit in the tile and ``1 <= num_runs <= MaxRuns``. Input keys are overwritten.
//! Keys must be default-constructible, copyable, and comparable by a strict weak ordering.
//! A block barrier is required before reusing the input or temporary storage after either output API.
//!
//! Example with sixteen shared-memory runs and blocked register output:
//!
//! .. code-block:: c++
//!
//!    using Merge = cub::BlockMerge<int, 128, 4, 16>;
//!    __shared__ Merge::TempStorage storage;
//!    __shared__ int keys[128 * 4 + 1];
//!    __shared__ int offsets[17];
//!    // Fill the sorted runs, offsets, and keys[offsets[16]].
//!    __syncthreads();
//!    int items[4];
//!    Merge(storage).Merge(keys, offsets, 16, items, cuda::std::less<>{});
//!    __syncthreads(); // Before reusing shared storage.
//!
//! @endrst
//!
//! @tparam KeyT Key type.
//! @tparam BlockDimX X dimension of the block.
//! @tparam ItemsPerThread Maximum number of output keys per thread.
//! @tparam MaxRuns Maximum number of sorted runs (1 through 32).
//! @tparam Algorithm Shared-memory strategy.
//! @tparam BlockDimY Y dimension of the block.
//! @tparam BlockDimZ Z dimension of the block.
template <typename KeyT,
          int BlockDimX,
          int ItemsPerThread,
          int MaxRuns                   = 32,
          BlockMergeAlgorithm Algorithm = BLOCK_MERGE_PING_PONG,
          int BlockDimY                 = 1,
          int BlockDimZ                 = 1>
class BlockMerge
{
  static constexpr int block_threads = BlockDimX * BlockDimY * BlockDimZ;
  static constexpr int tile_size     = block_threads * ItemsPerThread;
  static_assert(BlockDimX > 0 && BlockDimY > 0 && BlockDimZ > 0 && block_threads <= 1024,
                "Block dimensions must specify 1 through 1024 threads");
  static_assert(ItemsPerThread > 0, "ItemsPerThread must be positive");
  static_assert(MaxRuns >= 1 && MaxRuns <= 32, "MaxRuns must be between 1 and 32");
  static_assert(Algorithm == BLOCK_MERGE_PING_PONG || Algorithm == BLOCK_MERGE_REGISTER_STAGED,
                "Unsupported BlockMerge algorithm");

  struct _TempStorage
  {
    KeyT keys[Algorithm == BLOCK_MERGE_PING_PONG ? tile_size + 1 : 1];
  };

  _TempStorage& temp_storage;
  const int linear_tid;

  _CCCL_DEVICE_API static _TempStorage& private_storage()
  {
    __shared__ Uninitialized<_TempStorage> storage;
    return storage.Alias();
  }

public:
  //! @brief Opaque shared temporary storage; may be aliased with other block primitives after a barrier.
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //! @brief Constructs the collective with internally allocated shared storage.
  _CCCL_DEVICE_API BlockMerge()
      : temp_storage(private_storage())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @brief Constructs the collective using caller-provided shared storage.
  _CCCL_DEVICE_API explicit BlockMerge(TempStorage& storage)
      : temp_storage(storage.Alias())
      , linear_tid(RowMajorTid(BlockDimX, BlockDimY, BlockDimZ))
  {}

  //! @brief Returns blocked register output. Ranks outside the live prefix are value-initialized.
  //! @param[in,out] keys Shared key buffer, including one initialized slack key after the live prefix.
  //! @param[in] offsets Shared run boundaries.
  //! @param[in] num_runs Number of sorted runs.
  //! @param[out] out Per-thread blocked output.
  //! @param[in] compare_op Strict weak ordering used to sort every input run.
  template <typename CompareOp>
  _CCCL_DEVICE_API void
  Merge(KeyT* keys, const int* offsets, int num_runs, KeyT (&out)[ItemsPerThread], CompareOp compare_op)
  {
    if constexpr (Algorithm == BLOCK_MERGE_REGISTER_STAGED)
    {
      merge_regs(keys, offsets, num_runs, out, compare_op);
      return;
    }
    const KeyT* res = MergeToShared(keys, offsets, num_runs, compare_op);
    const int total = offsets[num_runs];
    const int base  = ItemsPerThread * linear_tid;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      out[i] = (base + i < total) ? res[base + i] : KeyT{};
    }
  }

  //! @brief Returns the shared buffer containing the merged live prefix.
  //! @param[in,out] keys Shared key buffer, including one initialized slack key after the live prefix.
  //! @param[in] offsets Shared run boundaries.
  //! @param[in] num_runs Number of sorted runs.
  //! @param[in] compare_op Strict weak ordering used to sort every input run.
  //! @return A pointer into keys or TempStorage, valid until those allocations are reused.
  template <typename CompareOp>
  [[nodiscard]] _CCCL_DEVICE_API const KeyT*
  MergeToShared(KeyT* keys, const int* offsets, int num_runs, CompareOp compare_op)
  {
    if constexpr (Algorithm == BLOCK_MERGE_REGISTER_STAGED)
    {
      KeyT out[ItemsPerThread];
      merge_regs(keys, offsets, num_runs, out, compare_op);
      __syncthreads(); // every thread has finished reading keys
      const int total = offsets[num_runs];
      const int base  = ItemsPerThread * linear_tid;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < ItemsPerThread; ++i)
      {
        if (base + i < total)
        {
          keys[base + i] = out[i];
        }
      }
      __syncthreads();
      return keys;
    }
    const int tid   = linear_tid;
    const int k     = num_runs;
    const int total = offsets[k];
    const int ke    = effective_runs(k);
    const int base  = ItemsPerThread * tid;
    const int end   = (::cuda::std::min) (base + ItemsPerThread, total);

    constexpr int pw_warps = block_threads / detail::warp_threads;
    constexpr bool pw_pow2 =
      block_threads % detail::warp_threads == 0 && pw_warps > 0 && (pw_warps & (pw_warps - 1)) == 0;
    const bool pair_warps = pw_pow2 && (ke & (ke - 1)) == 0 && ke <= 2 * pw_warps;
    const int pw_lgk      = 31 - __clz(ke);

    KeyT* buf_in  = keys;
    KeyT* buf_out = temp_storage.keys;
    if (linear_tid == 0)
    {
      buf_out[total] = KeyT{};
    }
    auto off_at = [&](int i) {
      return offsets[(::cuda::std::min) (i, k)];
    };

    for (int s = 1, r = 0; s < ke; s *= 2, ++r)
    {
      const int live      = (k + s - 1) >> r;
      const int num_pairs = (live + 1) >> 1;
      bool pw_done        = false;

      if (pair_warps)
      {
        constexpr int cap = ItemsPerThread + 2; // the odd surplus split hands out IPT + 2
        constexpr int lgw =
          Log2<(block_threads >= detail::warp_threads ? block_threads / detail::warp_threads : 1)>::VALUE;
        const int lg_nthr  = lgw - (pw_lgk - r - 1) + 5; // log2(threads of a pair)
        const int n_thr    = 1 << lg_nthr;
        const int p        = tid >> lg_nthr;
        const int lt       = tid & (n_thr - 1);
        const int a        = p << (r + 1);
        const int beg1     = off_at(a); // runs >= k are the empty padding runs
        const int beg2     = off_at(a + s);
        const int pair_end = off_at(a + 2 * s);
        const int n_pair   = pair_end - beg1;
        int lo, hi;
        window(n_pair, lg_nthr, lt, lo, hi);
        if (lo < hi)
        {
          const int n_out     = hi - lo;
          const int n1        = beg2 - beg1;
          const int n2        = pair_end - beg2;
          const int keys1_beg = detail::block_merge_path(buf_in + beg1, buf_in + beg2, n1, n2, lo, compare_op);

          int i1         = beg1 + keys1_beg;
          int i2         = beg2 + (lo - keys1_beg);
          const int end1 = beg2;
          const int end2 = pair_end;
          KeyT key1      = buf_in[i1];
          KeyT key2      = buf_in[i2];
          const int wpos = beg1 + lo;
          auto step      = [&](int item) {
            const bool take2     = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
            buf_out[wpos + item] = take2 ? key2 : key1;
            if (take2)
            {
              key2 = buf_in[++i2];
            }
            else
            {
              key1 = buf_in[++i1];
            }
          };
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int item = 0; item < cap; ++item)
          {
            if (item < n_out)
            {
              step(item);
            }
          }
          for (int item = cap; item < n_out; ++item) // a pair over cap * threads keys: rolled tail
          {
            step(item);
          }
        }
        pw_done = true;
      }

      int pos = pw_done ? end : base;
      while (pos < end)
      {
        const int p          = find_run(pos, offsets, num_pairs, 2 * s);
        const int a          = 2 * p * s;
        const int a_end      = (::cuda::std::min) (a + s, k);
        const int b_end      = (::cuda::std::min) (a + 2 * s, k);
        const int beg1       = offsets[a];
        const int beg2       = offsets[a_end];
        const int pair_end   = offsets[b_end];
        const int local_diag = pos - beg1;
        const int n_out      = (::cuda::std::min) (end, pair_end) - pos;
        const int keys1_beg =
          detail::block_merge_path(buf_in + beg1, buf_in + beg2, beg2 - beg1, pair_end - beg2, local_diag, compare_op);

        int i1         = beg1 + keys1_beg;
        int i2         = beg2 + (local_diag - keys1_beg);
        const int end1 = beg2;
        const int end2 = pair_end;
        // Refills are unconditional: i1 <= end1 <= total and i2 <= end2 <= total, and the buffers have N + 1 slots, so
        // the read one past either run is in bounds and its value is never used (the predicate tests the bound first).
        KeyT key1 = buf_in[i1];
        KeyT key2 = buf_in[i2];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < ItemsPerThread; ++item)
        {
          if (item < n_out)
          {
            const bool take2    = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
            buf_out[pos + item] = take2 ? key2 : key1;
            if (take2)
            {
              key2 = buf_in[++i2];
            }
            else
            {
              key1 = buf_in[++i1];
            }
          }
        }
        pos += n_out;
      }
      __syncthreads(); // disjoint writes into buf_out from a stable buf_in: one barrier per round
      KeyT* tmp = buf_in;
      buf_in    = buf_out;
      buf_out   = tmp;
    }
    return buf_in;
  }

private:
  template <typename CompareOp>
  _CCCL_DEVICE_API void
  merge_regs(KeyT* buf, const int* offsets, int k, KeyT (&out)[ItemsPerThread], CompareOp compare_op)
  {
    const int tid   = linear_tid;
    const int total = offsets[k];
    const int ke    = effective_runs(k);
    const int base  = ItemsPerThread * tid;
    const int end   = (::cuda::std::min) (base + ItemsPerThread, total);
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      out[i] = KeyT{};
    }
    constexpr int pw_warps = block_threads / detail::warp_threads;
    constexpr bool pw_pow2 =
      block_threads % detail::warp_threads == 0 && pw_warps > 0 && (pw_warps & (pw_warps - 1)) == 0;
    constexpr int lgw = Log2<(block_threads >= detail::warp_threads ? block_threads / detail::warp_threads : 1)>::VALUE;
    int pw_mask       = (pw_pow2 && k > 1 && (ke & (ke - 1)) == 0 && ke <= 2 * pw_warps) ? -1 : 0;
    const int pw_lgk  = 31 - __clz(ke);
    if (pw_mask != 0)
    {
      // fit bit r: every pair of round r has at most (IPT + 1) * n_thr keys (lane p holds offsets[min(p, k)]; pair p of
      // round r spans lanes p << (r + 1) .. (p + 1) << (r + 1)). The last round (r = pw_lgk - 1) is the plain one and
      // needs no bit.
      const int lane = tid & 31;
      const int o    = offsets[(::cuda::std::min) (lane, k)];
      int fit        = 0;
      for (int r = 0; r < pw_lgk - 1; ++r)
      {
        const int pairs = ke >> (r + 1);
        const int i0    = lane << (r + 1);
        const int i1    = (lane + 1) << (r + 1);
        const int b     = __shfl_sync(0xffffffffu, o, i0 & 31);
        const int e0    = __shfl_sync(0xffffffffu, o, i1 & 31);
        const int e     = i1 >= 32 ? total : e0;
        int len         = lane < pairs ? e - b : 0;
        for (int d = 16; d > 0; d >>= 1)
        {
          len = (::cuda::std::max) (len, __shfl_xor_sync(0xffffffffu, len, d));
        }
        const int n_thr = 1 << (lgw - (pw_lgk - r - 1) + 5);
        fit |= (len <= (ItemsPerThread + 1) * n_thr) ? (1 << r) : 0;
      }
      pw_mask = fit;
    }

    bool in_regs = false;
    for (int s = 1, r = 0; s < ke; s *= 2, ++r)
    {
      const bool last     = 2 * s >= ke;
      const int live      = (k + s - 1) >> r;
      const int num_pairs = (live + 1) >> 1;
      constexpr int cap   = ItemsPerThread + 2;
      KeyT pk[cap];
      int pw_pos   = 0;
      int pw_cnt   = 0;
      bool pw_done = false;

      if (((pw_mask >> r) & 1) && !last)
      {
        const int lg_nthr  = lgw - (pw_lgk - r - 1) + 5;
        const int p        = tid >> lg_nthr;
        const int lt       = tid & ((1 << lg_nthr) - 1);
        const int a        = p << (r + 1);
        const int beg1     = offsets[(::cuda::std::min) (a, k)];
        const int beg2     = offsets[(::cuda::std::min) (a + s, k)];
        const int pair_end = offsets[(::cuda::std::min) (a + 2 * s, k)];
        const int n_pair   = pair_end - beg1;
        int lo, hi;
        window(n_pair, lg_nthr, lt, lo, hi);
        pw_cnt = hi - lo; // <= IPT + 2 by the fit bit
        pw_pos = beg1 + lo;
        if (pw_cnt > 0)
        {
          const int n1        = beg2 - beg1;
          const int n2        = pair_end - beg2;
          const int keys1_beg = detail::block_merge_path(buf + beg1, buf + beg2, n1, n2, lo, compare_op);

          int i1         = beg1 + keys1_beg;
          int i2         = beg2 + (lo - keys1_beg);
          const int end1 = beg2;
          const int end2 = pair_end;
          KeyT key1      = buf[i1];
          KeyT key2      = buf[i2];
          _CCCL_PRAGMA_UNROLL_FULL()
          for (int item = 0; item < cap; ++item)
          {
            if (item < pw_cnt)
            {
              const bool take2 = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
              pk[item]         = take2 ? key2 : key1;
              if (take2)
              {
                key2 = buf[++i2];
              }
              else
              {
                key1 = buf[++i1];
              }
            }
          }
        }
        pw_done = true;
      }

      int pos_next = pw_done ? end : base;
      while (pos_next < end)
      {
        const int p          = find_run(pos_next, offsets, num_pairs, 2 * s);
        const int a          = 2 * p * s;
        const int a_end      = (::cuda::std::min) (a + s, k);
        const int b_end      = (::cuda::std::min) (a + 2 * s, k);
        const int beg1       = offsets[a];
        const int beg2       = offsets[a_end];
        const int pair_end   = offsets[b_end];
        const int pos_stop   = (::cuda::std::min) (end, pair_end);
        const int local_diag = pos_next - beg1;
        const int keys1_beg =
          detail::block_merge_path(buf + beg1, buf + beg2, beg2 - beg1, pair_end - beg2, local_diag, compare_op);

        int i1         = beg1 + keys1_beg;
        int i2         = beg2 + (local_diag - keys1_beg);
        const int end1 = beg2;
        const int end2 = pair_end;
        KeyT key1      = buf[i1];
        KeyT key2      = buf[i2];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < ItemsPerThread; ++item)
        {
          const int pos = base + item;
          if (pos >= pos_next && pos < pos_stop)
          {
            const bool take2 = (i2 < end2) && ((i1 >= end1) || compare_op(key2, key1));
            out[item]        = take2 ? key2 : key1;
            if (take2)
            {
              key2 = buf[++i2];
            }
            else
            {
              key1 = buf[++i1];
            }
          }
        }
        pos_next = pos_stop;
      }
      if (last)
      {
        in_regs = true;
        break;
      }
      __syncthreads(); // everyone has finished reading buf
      if (pw_done)
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < cap; ++item)
        {
          if (item < pw_cnt)
          {
            buf[pw_pos + item] = pk[item];
          }
        }
      }
      else
      {
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < ItemsPerThread; ++item)
        {
          if (base + item < end)
          {
            buf[base + item] = out[item];
          }
        }
      }
      __syncthreads(); // written back: the next round may read
    }
    if (!in_regs)
    {
      // k == 1 (nothing to merge): the loaded tile is the result
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        if (base + item < end)
        {
          out[item] = buf[base + item];
        }
      }
    }
  }

  // Padding enables the fixed-warp path without changing the real run boundaries.
  [[nodiscard]] _CCCL_DEVICE_API static int effective_runs(int num_runs)
  {
    constexpr int warps       = block_threads / detail::warp_threads;
    constexpr bool full_warps = block_threads % detail::warp_threads == 0 && warps > 0 && (warps & (warps - 1)) == 0;
    if (num_runs <= 2)
    {
      return num_runs;
    }
    const int padded = 1 << (32 - __clz(num_runs - 1));
    // A lone unmatched run does not benefit from padding.
    const bool pad = full_warps && (num_runs & (num_runs - 1)) != 0 && ((num_runs - 1) & (num_runs - 2)) != 0
                  && padded <= 2 * warps;
    return pad ? padded : num_runs;
  }

  //! Window [lo, hi) of the n outputs of one group for lane lt of nthr: the odd-stride split of the fixed-warps rounds.
  _CCCL_DEVICE_API static void window(int n, int lg_nthr, int lt, int& lo, int& hi)
  {
    const int nthr = 1 << lg_nthr;
    if (n <= ItemsPerThread * nthr)
    {
      lo = (::cuda::std::min) (ItemsPerThread * lt, n);
      hi = (::cuda::std::min) (lo + ItemsPerThread, n);
    }
    else if (n <= (ItemsPerThread + 1) * nthr)
    {
      const int rem = n - ItemsPerThread * nthr;
      const int x   = rem >> 1;
      const int odd = rem & 1;
      lo            = ItemsPerThread * lt + 2 * (::cuda::std::min) (lt, x) + (lt > x ? odd : 0);
      hi            = lo + ItemsPerThread + (lt < x ? 2 : (lt == x ? odd : 0));
    }
    else
    {
      const int q   = n >> lg_nthr;
      const int b   = (q - 1) | 1;
      const int rem = n - b * nthr;
      const int x   = rem >> 1;
      const int odd = rem & 1;
      lo            = b * lt + 2 * (::cuda::std::min) (lt, x) + (lt > x ? odd : 0);
      hi            = lo + b + (lt < x ? 2 : (lt == x ? odd : 0));
    }
  }

  //! Largest index i in [0, count) with off[i * stride] <= idx (off non-decreasing).
  _CCCL_DEVICE_API static int find_run(int idx, const int* off, int count, int stride)
  {
    int lo = 0;
    int hi = count;
    while (hi - lo > 1)
    {
      const int mid = lo + (hi - lo) / 2;
      if (off[mid * stride] <= idx)
      {
        lo = mid;
      }
      else
      {
        hi = mid;
      }
    }
    return lo;
  }
};

CUB_NAMESPACE_END
