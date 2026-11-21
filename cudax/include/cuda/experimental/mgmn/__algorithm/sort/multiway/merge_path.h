// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_MERGE_PATH_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_MERGE_PATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
#  include <cub/util_math.cuh> // CUB_NS_QUALIFIER::MidPoint
#  include <cub/util_type.cuh> // CUB_NS_QUALIFIER::detail::it_value_t

#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__algorithm/min.h>

#  include <cuda/experimental/mgmn/__algorithm/common.h>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
namespace __multiway
{
//! Computes the merge-path intersection for diagonal `diag` (Odeh et al., "Merge Path - Parallel Merging Made Simple").
//! Returns the number of elements consumed from keys1; `diag - result` elements are consumed from keys2.
//! Precondition: diag <= keys1_count + keys2_count.
template <typename KeyIt1, typename KeyIt2, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT
merge_path(KeyIt1 keys1, KeyIt2 keys2, OffsetT keys1_count, OffsetT keys2_count, OffsetT diag, BinaryPred binary_pred)
{
  OffsetT keys1_begin = diag < keys2_count ? 0 : diag - keys2_count;
  OffsetT keys1_end   = (::cuda::std::min) (diag, keys1_count);

  while (keys1_begin < keys1_end)
  {
    const OffsetT mid = CUB_NS_QUALIFIER::MidPoint<OffsetT>(keys1_begin, keys1_end);
    // pull copies of the keys before calling binary_pred so proxy references are unwrapped
    const CUB_NS_QUALIFIER::detail::it_value_t<KeyIt1> key1 = keys1[mid];
    const CUB_NS_QUALIFIER::detail::it_value_t<KeyIt2> key2 = keys2[diag - 1 - mid];
    if (binary_pred(key2, key1))
    {
      keys1_end = mid;
    }
    else
    {
      keys1_begin = mid + 1;
    }
  }
  return keys1_begin;
}

//! merge_path that starts from a guess. `est` is a predicted crossing (the agent uses the proportional
//! estimate diag * n1 / (n1 + n2)): one probe at the guess gives the side, then `Probes` probes at
//! geometrically growing distances (`First`, 4 First, 16 First, ...) from it bracket the crossing, and a
//! plain bisection finishes inside the bracket. Returns exactly what merge_path returns. On random input
//! the crossing sits within ~16 of the guess in round 1 and ~32 in round 4 of the in-SM merge tree
//! (SESSION_2026-10-20_straddle.md §14), so this costs ~6-9 dependent steps where the bisection from the
//! full bracket costs 10-12. `far` is set when all `Probes` probes were spent without bracketing (the
//! guess was off by more than First * 4^(Probes-1)): the caller then pays gallop + bisection, and uses
//! the flag to switch the warp back to the plain search for the next rounds.
template <int First, int Probes, typename KeyIt1, typename KeyIt2, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE int merge_path_gallop(
  KeyIt1 keys1, KeyIt2 keys2, int keys1_count, int keys2_count, int diag, int est, BinaryPred binary_pred, bool& far)
{
  static_assert(First > 0 && Probes >= 1, "merge_path_gallop: First > 0, Probes >= 1");
  int lo = diag < keys2_count ? 0 : diag - keys2_count;
  int hi = (::cuda::std::min) (diag, keys1_count);
  far    = false;
  if (lo < hi)
  {
    // The predicate keys2[diag - 1 - i] < keys1[i] is false up to the crossing and true from it on; every
    // index probed here is in [lo, hi), the same range the bisection probes.
    const int c = (::cuda::std::min) ((::cuda::std::max) (est, lo), hi - 1);
    if (binary_pred(keys2[diag - 1 - c], keys1[c]))
    {
      hi       = c; // crossing <= c: gallop down
      int step = First;
      int g    = 0;
      for (; g < Probes; ++g)
      {
        const int q = c - step;
        if (q < lo)
        {
          break; // the bracket is already inside the gallop window
        }
        if (binary_pred(keys2[diag - 1 - q], keys1[q]))
        {
          hi = q;
        }
        else
        {
          lo = q + 1;
          break;
        }
        step *= 4;
      }
      far = g == Probes;
    }
    else
    {
      lo       = c + 1; // crossing > c: gallop up
      int step = First;
      int g    = 0;
      for (; g < Probes; ++g)
      {
        const int q = c + step;
        if (q >= hi)
        {
          break;
        }
        if (binary_pred(keys2[diag - 1 - q], keys1[q]))
        {
          hi = q;
          break;
        }
        lo = q + 1;
        step *= 4;
      }
      far = g == Probes;
    }
    while (lo < hi)
    {
      const int mid = CUB_NS_QUALIFIER::MidPoint<int>(lo, hi);
      if (binary_pred(keys2[diag - 1 - mid], keys1[mid]))
      {
        hi = mid;
      }
      else
      {
        lo = mid + 1;
      }
    }
  }
  return lo;
}

//! merge_path with `Probes` (1, 3 or 7) independent probes per step: the merge-path predicate
//! (keys2[diag - 1 - mid] < keys1[mid]) is monotone in mid, so the number of probes where it is
//! false gives the new bracket directly, and a step costs 2 x Probes independent shared loads instead
//! of 2 dependent ones while cutting the dependent chain from log2 to log(Probes + 1) steps
//! (NOTES_K_IPT_L2.md §9.14: the two-buffer round on the B200 is bound by that chain, not by
//! instructions). Probes == 1 is the plain merge_path above.
template <int Probes, typename KeyIt1, typename KeyIt2, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT merge_path_probes(
  KeyIt1 keys1, KeyIt2 keys2, OffsetT keys1_count, OffsetT keys2_count, OffsetT diag, BinaryPred binary_pred)
{
  static_assert(Probes == 1 || Probes == 3 || Probes == 7, "Probes + 1 must be a power of two");
  if constexpr (Probes == 1)
  {
    return merge_path(keys1, keys2, keys1_count, keys2_count, diag, binary_pred);
  }
  else
  {
    OffsetT lo = diag < keys2_count ? 0 : diag - keys2_count;
    OffsetT hi = (::cuda::std::min) (diag, keys1_count);
    while (hi - lo > Probes)
    {
      const OffsetT step = (hi - lo) / (Probes + 1); // >= 1 here; the probes are strictly increasing
      OffsetT mids[Probes];
      bool below[Probes]; // predicate false: the split is past this probe
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < Probes; ++j)
      {
        mids[j] = lo + step * static_cast<OffsetT>(j + 1);
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < Probes; ++j)
      {
        const CUB_NS_QUALIFIER::detail::it_value_t<KeyIt1> key1 = keys1[mids[j]];
        const CUB_NS_QUALIFIER::detail::it_value_t<KeyIt2> key2 = keys2[diag - 1 - mids[j]];
        below[j]                                                = !binary_pred(key2, key1);
      }
      OffsetT new_lo = lo, new_hi = hi;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int j = 0; j < Probes; ++j)
      {
        new_lo = below[j] ? mids[j] + 1 : new_lo; // the last probe with the predicate false
        new_hi = (!below[j] && new_hi == hi) ? mids[j] : new_hi; // the first probe with it true
      }
      lo = new_lo;
      hi = new_hi;
    }
    while (lo < hi) // at most Probes elements left: plain bisection
    {
      const OffsetT mid                                       = CUB_NS_QUALIFIER::MidPoint<OffsetT>(lo, hi);
      const CUB_NS_QUALIFIER::detail::it_value_t<KeyIt1> key1 = keys1[mid];
      const CUB_NS_QUALIFIER::detail::it_value_t<KeyIt2> key2 = keys2[diag - 1 - mid];
      if (binary_pred(key2, key1))
      {
        hi = mid;
      }
      else
      {
        lo = mid + 1;
      }
    }
    return lo;
  }
}

//! Merges ItemsPerThread elements from two runs that both live in `keys_shared`.
//! keys1_count/keys2_count bound how many elements may be consumed from each run. One element past the end of each
//! run may be *read* (never used), so the shared buffer must have one element of slack.
//! `indices[i]` receives the shared-memory index the i-th output element came from (used to gather values later).
template <bool Unroll, typename KeyIt, typename KeyT, typename CompareOp, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void serial_merge(
  KeyIt keys_shared,
  int keys1_beg,
  int keys2_beg,
  int keys1_count,
  int keys2_count,
  KeyT (&output)[ItemsPerThread],
  int (&indices)[ItemsPerThread],
  CompareOp compare_op,
  KeyT oob_default)
{
  const int keys1_end = keys1_beg + keys1_count;
  const int keys2_end = keys2_beg + keys2_count;

  KeyT key1 = keys1_count != 0 ? keys_shared[keys1_beg] : oob_default;
  KeyT key2 = keys2_count != 0 ? keys_shared[keys2_beg] : oob_default;

  _CCCL_PRAGMA_UNROLL(Unroll ? ItemsPerThread : 1)
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const bool p  = (keys2_beg < keys2_end) && ((keys1_beg >= keys1_end) || compare_op(key2, key1));
    output[item]  = p ? key2 : key1;
    indices[item] = p ? keys2_beg++ : keys1_beg++;
    if (p)
    {
      key2 = keys_shared[keys2_beg];
    }
    else
    {
      key1 = keys_shared[keys1_beg];
    }
  }
}

template <bool Unroll, typename KeyIt, typename KeyT, typename CompareOp, int ItemsPerThread>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void serial_merge(
  KeyIt keys_shared,
  int keys1_beg,
  int keys2_beg,
  int keys1_count,
  int keys2_count,
  KeyT (&output)[ItemsPerThread],
  int (&indices)[ItemsPerThread],
  CompareOp compare_op)
{
  serial_merge<Unroll>(
    keys_shared, keys1_beg, keys2_beg, keys1_count, keys2_count, output, indices, compare_op, output[0]);
}

//! Loads two consecutive global ranges (count1 elements from input1, then count2 from input2) into registers in a
//! coalesced, block-strided fashion: output[item] <- concat(input1, input2)[BlockThreads * item + threadIdx.x].
template <int BlockThreads, bool IsFullTile, int ItemsPerThread, class T, class It1, class It2>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
gmem_to_reg(T (&output)[ItemsPerThread], It1 input1, It2 input2, int count1, int count2)
{
  if constexpr (IsFullTile)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int idx = BlockThreads * item + static_cast<int>(threadIdx.x);
      // It1 and It2 could have different value types. Convert after load.
      output[item] = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int idx = BlockThreads * item + static_cast<int>(threadIdx.x);
      if (idx < count1 + count2)
      {
        output[item] = (idx < count1) ? static_cast<T>(input1[idx]) : static_cast<T>(input2[idx - count1]);
      }
    }
  }
}

//! Stores registers to shared memory in a coalesced fashion: out[BlockThreads * item + tid] <- in[item].
template <int BlockThreads, int ItemsPerThread, class T, class It>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void reg_to_shared(It output, T (&input)[ItemsPerThread])
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    const int idx = BlockThreads * item + static_cast<int>(threadIdx.x);
    output[idx]   = input[item];
  }
}
} // namespace __multiway
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILER(NVCC)

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_MERGE_PATH_H
