// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_PARTITION_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_PARTITION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
#  include <cuda/std/cstdint>

#  include <cuda/experimental/mgmn/__algorithm/common.h>

#  include <cstring>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
namespace __multiway
{
//! Proportional first-round pivots of the multiselect (see mway_select_ranged): used while the leader's window holds at
//! least mway_prop_min keys; the pivot offset is mway_prop_mult * sqrt(window); the pivot index is snapped to a grid of
//! 2^(floor(log2(offset)) - mway_prop_qshift) so neighbouring diagonals load the same element.
inline constexpr int mway_prop_min    = 4096;
inline constexpr int mway_prop_qshift = 2;
inline constexpr float mway_prop_mult = 2.0f;

#  ifdef _CUDAX_SORT_MWAY_MWAY_COUNT_ROUNDS
//! Diagnostic only (compile with -D_CUDAX_SORT_MWAY_MWAY_COUNT_ROUNDS): multiselect searches and rounds,
//! summed over all warps of all launches since the last reset (the driver zeroes them, runs one
//! sort, and prints per-search figures). `probe` is an estimate of the dependent global loads a
//! round costs: per live lane, one binary search of ceil(log2(range)) steps plus two gallop probes.
__device__ unsigned long long mway_search_counter     = 0; // mway_select_ranged calls (one per diagonal per launch)
__device__ unsigned long long mway_round_counter      = 0; // warp rounds
__device__ unsigned long long mway_lane_round_counter = 0; // live lanes summed over rounds
__device__ unsigned long long mway_probe_counter      = 0; // estimated binary-search probes (global loads)
#  endif

//! Warp-uniform shuffle for arbitrary trivially-copyable types up to 8 bytes (covers every key
//! type this repo tests: u8/i16/i32/u32/i64/f32/f64), via reinterpretation through the native
//! 32-/64-bit shuffle intrinsics. __shfl_sync's built-in overload set doesn't cover narrow integer
//! types directly, so this is the one place that needs the generic byte-copy detour.
template <typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T mway_shfl_sync(unsigned mask, T value, int src_lane)
{
  static_assert(sizeof(T) <= 8, "mway_shfl_sync supports types up to 8 bytes");
  if constexpr (sizeof(T) <= 4)
  {
    unsigned int bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = __shfl_sync(mask, bits, src_lane);
    T out{};
    std::memcpy(&out, &bits, sizeof(T));
    return out;
  }
  else
  {
    unsigned long long bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = __shfl_sync(mask, bits, src_lane);
    T out{};
    std::memcpy(&out, &bits, sizeof(T));
    return out;
  }
}

template <typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T mway_shfl_xor_sync(unsigned mask, T value, int lane_mask)
{
  static_assert(sizeof(T) <= 8, "mway_shfl_xor_sync supports types up to 8 bytes");
  if constexpr (sizeof(T) <= 4)
  {
    unsigned int bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = __shfl_xor_sync(mask, bits, lane_mask);
    T out{};
    std::memcpy(&out, &bits, sizeof(T));
    return out;
  }
  else
  {
    unsigned long long bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = __shfl_xor_sync(mask, bits, lane_mask);
    T out{};
    std::memcpy(&out, &bits, sizeof(T));
    return out;
  }
}

template <typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T mway_shfl_up_sync(unsigned mask, T value, unsigned delta)
{
  static_assert(sizeof(T) <= 8, "mway_shfl_up_sync supports types up to 8 bytes");
  if constexpr (sizeof(T) <= 4)
  {
    unsigned int bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = __shfl_up_sync(mask, bits, delta);
    T out{};
    std::memcpy(&out, &bits, sizeof(T));
    return out;
  }
  else
  {
    unsigned long long bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    bits = __shfl_up_sync(mask, bits, delta);
    T out{};
    std::memcpy(&out, &bits, sizeof(T));
    return out;
  }
}

//! Full-warp sum reduction (every lane receives the total). Lanes outside `mask` must pass 0.
template <typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T mway_warp_reduce_add(unsigned mask, T value)
{
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int offset = 16; offset > 0; offset >>= 1)
  {
    value += mway_shfl_xor_sync(mask, value, offset);
  }
  return value;
}

//! Warp-exclusive prefix sum: lane L receives the sum of `value` over lanes [0, L).
template <typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE T mway_warp_exclusive_scan_add(unsigned mask, T value)
{
  const int lane = static_cast<int>(threadIdx.x & 31u);
  T inclusive    = value;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int offset = 1; offset < 32; offset <<= 1)
  {
    const T other = mway_shfl_up_sync(mask, inclusive, static_cast<unsigned>(offset));
    if (lane >= offset)
    {
      inclusive += other;
    }
  }
  return inclusive - value;
}

// ---------------------------------------------------------------------------------------------
// Sector-aware search tail
// ---------------------------------------------------------------------------------------------
//
// A binary search over global memory is a chain of dependent loads, each of which fetches a whole
// 32-byte sector to use one key. Its last levels operate inside a window of a few keys that sits
// in one or two sectors anyway. The tail below replaces those levels: once the range holds at most
// `mway_tail_keys` keys (32 bytes' worth: 8 x int32, 4 x int64), it loads the 16-byte chunks that
// cover the range with independent vector loads issued together, then counts in registers how many
// keys satisfy the ordering test. Because the range is sorted, that count IS the bound. Three
// dependent round trips become one, and the sector count drops as well (the chunks fetched are
// the sectors the scalar probes would have touched, fetched once).
//
// Chunks are the 16-byte-aligned pieces from align_down(&array[first]) to align_up(&array[last]),
// so every chunk loaded contains at least one element of [first, last): the reads never leave the
// allocation as long as it has >= 16-byte granularity (cudaMalloc gives 256). Keys are extracted
// with memcpy, so KeyT only has to be trivially copyable with a size dividing 16.
//
// Measured 2026-09-21 (NOTES_K_IPT_L2.md §6.9): neutral within 1 % on the RTX 3090, RTX 5090 and
// L40S at 64M and 256M, with the L2 sector count unchanged (a 16-byte load is half a sector, so
// the chunk loads cost as many requests as the probes they replace) and 14 more registers in the
// partition kernel. Opt-in with -D_CUDAX_SORT_MWAY_MWAY_SECTOR_TAIL; the plain binary search is the
// default.

inline constexpr int mway_tail_chunk_bytes = 16;

template <typename KeyT>
inline constexpr int mway_tail_keys = 2 * (mway_tail_chunk_bytes / static_cast<int>(sizeof(KeyT)));

//! Number of elements of the sorted range [first, last) (at most mway_tail_keys<KeyT>, up to three
//! 16-byte chunks) for which `test(element)` is true, where `test` is monotone (true then false).
template <typename KeyT, typename OffsetT, typename Test>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT mway_tail_count(const KeyT* array, OffsetT first, OffsetT last, Test test)
{
  static_assert(mway_tail_chunk_bytes % sizeof(KeyT) == 0, "key size must divide 16 bytes");
  constexpr int keys_per_chunk = mway_tail_chunk_bytes / static_cast<int>(sizeof(KeyT));
  constexpr int max_chunks     = 3; // <= 32 bytes of keys span at most three 16-byte chunks
  constexpr int max_keys       = max_chunks * keys_per_chunk;

  const auto p_first = reinterpret_cast<::cuda::std::uintptr_t>(array + first);
  const auto p_end   = reinterpret_cast<::cuda::std::uintptr_t>(array + last); // one past the last element
  const auto a0      = p_first & ~static_cast<::cuda::std::uintptr_t>(mway_tail_chunk_bytes - 1);
  const auto a1 =
    (p_end + (mway_tail_chunk_bytes - 1)) & ~static_cast<::cuda::std::uintptr_t>(mway_tail_chunk_bytes - 1);
  const int num_chunks = static_cast<int>((a1 - a0) / mway_tail_chunk_bytes); // 1..3
  const int skip       = static_cast<int>((p_first - a0) / sizeof(KeyT)); // keys in chunk 0 before `first`
  const int count      = static_cast<int>(last - first);

  // Independent loads: all issue before any completes.
  uint4 chunks[max_chunks];
  const uint4* src = reinterpret_cast<const uint4*>(a0);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int c = 0; c < max_chunks; ++c)
  {
    chunks[c] = (c < num_chunks) ? src[c] : uint4{0u, 0u, 0u, 0u};
  }

  const char* bytes = reinterpret_cast<const char*>(chunks);
  int hits          = 0;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int j = 0; j < max_keys; ++j)
  {
    const int i = j - skip; // index relative to `first`
    if (i >= 0 && i < count)
    {
      KeyT k;
      std::memcpy(&k, bytes + j * sizeof(KeyT), sizeof(KeyT));
      hits += test(k) ? 1 : 0;
    }
  }
  return static_cast<OffsetT>(hits);
}

// ---------------------------------------------------------------------------------------------
// k-ary probes
// ---------------------------------------------------------------------------------------------
//
// The multiselect's per-round lower-bound search is a chain of dependent global loads, one per
// halving, and the partition kernel is bound by the length of that chain, not by the number of
// searches in flight (NOTES_K_IPT_L2.md §6.8, §9.8). A step that loads M evenly spaced keys at once
// (independent loads, issued back to back, one round trip) narrows the range by M + 1 instead of 2,
// so the chain is log2(M + 1) times shorter at M / log2(M + 1) times the loads (1.5x at M = 3, 2.3x
// at M = 7). The probes below `key` are a prefix of the M, so the new range follows from their count
// alone: no dynamically indexed array (which would go to local memory) and no data-dependent branch.
// M = 1 is the plain binary search. M is a template parameter of the search and of the partition
// kernel; the mergesort picks it per launch (mway_sort.cuh, mway_partition_probes): the wide search
// wins while the launch is latency-bound (few chains in flight) and loses once it is bound by
// memory-request throughput (a 32-way pass over 2^28 keys keeps a million uncoalesced chains in
// flight and its time tracks the request count: M = 3 +21 %, M = 7 +87 % on the B200). Compile with
// -D_CUDAX_SORT_MWAY_MWAY_PROBES=1|3|7 to force one M everywhere (A/B builds).
#  if defined(_CUDAX_SORT_MWAY_MWAY_PROBES)
inline constexpr int mway_forced_probes = _CUDAX_SORT_MWAY_MWAY_PROBES;
static_assert(mway_forced_probes == 1 || mway_forced_probes == 3 || mway_forced_probes == 7,
              "_CUDAX_SORT_MWAY_MWAY_PROBES must be 1, 3 or 7 (the instantiated variants)");
#  else
inline constexpr int mway_forced_probes = 0; // 0: choose per launch
#  endif

//! Narrows [first, last) to at most `Probes` elements around lower_bound(key) with (Probes + 1)-ary
//! steps. Postcondition: lower_bound(key) over the original range lies in the returned [first, last).
template <int Probes, typename KeyT, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
mway_narrow_lower_bound(const KeyT* array, OffsetT& first, OffsetT& last, KeyT key, BinaryPred binary_pred)
{
  while (last - first > static_cast<OffsetT>(Probes))
  {
    const OffsetT span = (last - first) / (Probes + 1); // >= 1: probes at first + (i + 1) * span
    KeyT probe[Probes];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < Probes; ++i)
    {
      probe[i] = array[first + static_cast<OffsetT>(i + 1) * span];
    }
    // Issue all probe loads before any compare: left to itself, ptxas issued the last probe load only after the first
    // compare (reusing its register), i.e. two memory round trips per iteration instead of one. The warp-wide barrier
    // pins the loads ahead of it (an empty asm on the values is optimised away).
    __syncwarp(__activemask());
    int below = 0; // probes ordered before key: a prefix of the sorted range
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < Probes; ++i)
    {
      below += binary_pred(probe[i], key) ? 1 : 0;
    }
    // lower_bound is in (probe[below - 1], probe[below]], i.e. [first + below * span + 1, first +
    // (below + 1) * span] with the outer probes standing in for first - 1 and last.
    const OffsetT lo = below > 0 ? first + static_cast<OffsetT>(below) * span + 1 : first;
    const OffsetT hi = below < Probes ? first + static_cast<OffsetT>(below + 1) * span + 1 : last;
    first            = lo;
    last             = hi;
  }
}

template <int Probes = 1, typename KeyT, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT
mway_lower_bound(const KeyT* array, OffsetT first, OffsetT last, KeyT key, BinaryPred binary_pred)
{
  if constexpr (Probes > 1)
  {
    mway_narrow_lower_bound<Probes>(array, first, last, key, binary_pred);
  }
#  ifdef _CUDAX_SORT_MWAY_MWAY_SECTOR_TAIL
  while (last - first > static_cast<OffsetT>(mway_tail_keys<KeyT>))
#  else
  while (first < last)
#  endif
  {
    const OffsetT mid = first + (last - first) / 2;
    if (binary_pred(array[mid], key))
    {
      first = mid + 1;
    }
    else
    {
      last = mid;
    }
  }
#  ifdef _CUDAX_SORT_MWAY_MWAY_SECTOR_TAIL
  if (first < last)
  {
    // elements < key form a prefix of the sorted range
    first += mway_tail_count(array, first, last, [&](const KeyT& k) {
      return binary_pred(k, key);
    });
  }
#  endif
  return first;
}

template <typename KeyT, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT
mway_upper_bound(const KeyT* array, OffsetT first, OffsetT last, KeyT key, BinaryPred binary_pred)
{
#  ifdef _CUDAX_SORT_MWAY_MWAY_SECTOR_TAIL
  while (last - first > static_cast<OffsetT>(mway_tail_keys<KeyT>))
#  else
  while (first < last)
#  endif
  {
    const OffsetT mid = first + (last - first) / 2;
    if (binary_pred(key, array[mid]))
    {
      last = mid;
    }
    else
    {
      first = mid + 1;
    }
  }
#  ifdef _CUDAX_SORT_MWAY_MWAY_SECTOR_TAIL
  if (first < last)
  {
    // elements <= key (i.e. !(key < element)) form a prefix of the sorted range
    first += mway_tail_count(array, first, last, [&](const KeyT& k) {
      return !binary_pred(key, k);
    });
  }
#  endif
  return first;
}

//! upper_bound(key) over [lower, last) given `lower == lower_bound(key)`, by galloping from
//! `lower`: probes lower, lower+1, lower+2, lower+4, ... until an element > key (or `last`) is
//! found, then binary-searches the last gap. Costs O(log(#equal keys)) probes -- ONE probe when
//! `key` has no duplicates -- instead of a full log2(last - first) binary search, halving the
//! multiselect's dominant cost (dependent global loads) on typical inputs.
template <typename KeyT, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT
mway_upper_bound_gallop(const KeyT* array, OffsetT lower, OffsetT last, KeyT key, BinaryPred binary_pred)
{
  OffsetT lo   = lower; // invariant: everything in [lower, lo) is equal to key
  OffsetT hi   = lower;
  OffsetT step = 1;
  while (true)
  {
    if (hi >= last)
    {
      hi = last;
      break;
    }
    if (binary_pred(key, array[hi]))
    {
      break; // array[hi] > key
    }
    lo = hi + 1;
    hi = lower + step;
    step <<= 1;
  }
  return mway_upper_bound(array, lo, hi, key, binary_pred);
}

// Keep temporary sums, doubled pivot weights and proportional-pivot overshoot
// representable in signed32, including optional non-default pivot rules.
inline constexpr int mway_relative32_max_span = 0x3fffffff;

// q2 snaps against the original run's absolute element grid. A rebased search
// must add its wide origin before snapping, then return to relative coordinates.
template <typename OffsetT, typename GridOffsetT>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE OffsetT
mway_snap_pivot_index(OffsetT idx, OffsetT first, OffsetT last, int shift, GridOffsetT origin)
{
  GridOffsetT absolute    = origin + static_cast<GridOffsetT>(idx);
  absolute                = (absolute >> shift) << shift;
  const GridOffsetT lower = origin + static_cast<GridOffsetT>(first);
  const GridOffsetT upper = origin + static_cast<GridOffsetT>(last);
  absolute                = absolute < lower ? lower : absolute;
  absolute                = absolute >= upper ? upper - 1 : absolute;
  return static_cast<OffsetT>(absolute - origin);
}

//! Warp-cooperative K-way multiselect. Every lane in [0, 32) must call this together (uniform
//! control flow across the warp). Lane `lane` owns array `array` of length `count` when
//! `lane < k` (`k <= 32`); lanes `>= k` must pass `array = nullptr`, `count = 0` and their return
//! value is unused by the caller.
//!
//! Returns this lane's cut offset: the number of elements of its own array that belong strictly
//! before global output rank `diag`. Summed over all k active lanes, the cuts equal `diag`
//! (mirrors `merge_path`'s contract for K=2, generalized to K arrays).
//! Warm-started variant: this lane's cut is already known to lie in [first, last), and `remaining`
//! is `diag` minus the sum of all lanes' `first` (the output rank still unaccounted for once
//! everything before every lane's `first` is taken). The plain overload below is the special case
//! first = 0, last = count, remaining = diag.
//!
//! The K-way mergesort uses this for a two-level search over long segments: cuts are monotone in
//! the diagonal (the algorithm resolves ties deterministically by lane order, so the merged order
//! is the total order (key, lane, index) and cut_i(d) is non-decreasing in d), hence the cuts of two
//! bracketing, already-computed diagonals bound the cuts of every diagonal between them. Starting
//! from that bracket instead of the whole segment shortens both the binary searches (log2 of the
//! bracket width instead of log2 of the segment) and the number of rounds.
//! `Probes`: keys loaded per step of each lane's lower-bound search (mway_narrow_lower_bound); the
//! cuts do not depend on it.
template <int Probes = 1, typename KeyT, typename OffsetT, typename BinaryPred, typename GridOffsetT = OffsetT>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT mway_select_ranged(
  const KeyT* array,
  OffsetT first,
  OffsetT last,
  int k,
  OffsetT remaining,
  BinaryPred binary_pred,
  unsigned mask           = 0xffffffffu,
  GridOffsetT grid_origin = GridOffsetT{0})
{
  const int lane = static_cast<int>(threadIdx.x & 31u);
  if (lane >= k)
  {
    first = 0;
    last  = 0;
  }
#  ifdef _CUDAX_SORT_MWAY_MWAY_COUNT_ROUNDS
  if (lane == 0)
  {
    atomicAdd(&mway_search_counter, 1ull);
  }
#  endif

  [[maybe_unused]] unsigned round_ix = 0;
  while (__ballot_sync(mask, first < last) != 0u)
  {
    ++round_ix;
    const unsigned live = __ballot_sync(mask, first < last);
    const bool active   = first < last;
#  ifdef _CUDAX_SORT_MWAY_MWAY_COUNT_ROUNDS
    {
      const OffsetT range   = active ? last - first : OffsetT{0};
      const int steps       = active ? (64 - __clzll(static_cast<unsigned long long>(range))) + 2 : 0;
      const int probes_warp = mway_warp_reduce_add(mask, steps);
      if (lane == 0)
      {
        atomicAdd(&mway_round_counter, 1ull);
        atomicAdd(&mway_lane_round_counter, static_cast<unsigned long long>(__popc(live)));
        atomicAdd(&mway_probe_counter, static_cast<unsigned long long>(probes_warp));
      }
    }
#  endif

    // Pivot selection. Three rules, selected at compile time (see NOTES_MWAY_SORT.md for the A/B):
    //
    //   default   LARGEST-RANGE: midpoint of the live lane with the largest remaining range. One
    //             warp max-reduction; only that lane loads its midpoint. Every round at least halves
    //             the largest range and kills every lane lying entirely on the far side of the
    //             pivot, so disjoint-range inputs (e.g. the runs of an already-sorted array)
    //             converge in ~log2(K) + log2(n) rounds instead of the leader rule's K * log2(n).
    //             Matches the leader rule's cost on random inputs.
    //   _CUDAX_SORT_MWAY_MWAY_WMEDIAN_PIVOT   weighted median of all live lanes' midpoints (weights = range
    //             lengths): provably discards >= 1/4 of all remaining candidates per round, the
    //             tightest bound (best on disjoint inputs) but ~30% slower on random inputs because
    //             every lane loads a midpoint and the median needs an O(K) shuffle loop per round.
    //   _CUDAX_SORT_MWAY_MWAY_LEADER_PIVOT    the original rule: lowest live lane's midpoint. Degenerates on
    //             disjoint ranges (resolves one lane at a time).
#  if defined(_CUDAX_SORT_MWAY_MWAY_RANDOM_PIVOT)
    // Weight-proportional random pivot (2026-09-28): every live lane loads its own midpoint (one round
    // trip, like the leader's single load), the warp prefix-sums the bracket lengths, a hash of
    // (remaining, round) picks a threshold in [0, total) and the lane whose prefix interval holds it
    // supplies the pivot -- randomized quickselect over the union of the brackets: a quarter of all
    // remaining candidates discarded per round in expectation, no pathology on disjoint or skewed
    // brackets, for ~12 shuffles instead of the exact weighted median's O(K) loop. The cuts do not
    // depend on the pivot, and the hash makes the rounds reproducible. Threshold by 32-bit
    // multiply-high; total_w is truncated to 32 bits (brackets are bounded by 32 tiles per lane after
    // the coarse launch, whole segments before it; a wrap only biases the draw, the threshold stays
    // below total_w). Measured 2026-09-28 (NOTES_K_IPT_L2.md §9.12): rounds between the two other
    // rules; on random data neutral on the L40S and B200 (+0.3 % of the sort), +1.4 % on the 5090
    // (every live lane loads a midpoint each round: ~10 % more probes); on sorted / reversed input
    // 13-36 % off the partition. Opt-in, like the weighted median: largest-range stays the default.
    const OffsetT weight   = active ? (last - first) : OffsetT{0};
    const OffsetT mid_here = first + (last - first) / 2;
    const KeyT key_here    = active ? array[mid_here] : KeyT{};
    const OffsetT excl     = mway_warp_exclusive_scan_add(mask, weight);
    const OffsetT total_w  = mway_shfl_sync(mask, excl + weight, 31);
    // Round index only, deliberately: hashing the diagonal's rank in as well made neighbouring
    // diagonals pick different pivot lanes, their probes stopped sharing sectors, and the partition
    // ran 50-110 % slower at the same round count (§9.12). Neighbours must stay in step.
    unsigned h = round_ix * 0x9E3779B9u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    const OffsetT thresh = static_cast<OffsetT>(__umulhi(h, static_cast<unsigned>(total_w)));
    const bool holds     = active && excl <= thresh && thresh < excl + weight;
    const int leader     = __ffs(static_cast<int>(__ballot_sync(mask, holds))) - 1;
    const KeyT pivot     = mway_shfl_sync(mask, key_here, leader);
#  elif defined(_CUDAX_SORT_MWAY_MWAY_LEADER_PIVOT)
    const int leader           = __ffs(static_cast<int>(live)) - 1;
    const OffsetT leader_first = mway_shfl_sync(mask, first, leader);
    const OffsetT leader_last  = mway_shfl_sync(mask, last, leader);
    const OffsetT mid          = leader_first + (leader_last - leader_first) / 2;
    const KeyT pivot_local     = (lane == leader) ? array[mid] : KeyT{};
    const KeyT pivot           = mway_shfl_sync(mask, pivot_local, leader);
#  elif defined(_CUDAX_SORT_MWAY_MWAY_WMEDIAN_PIVOT)
    const OffsetT weight   = active ? (last - first) : OffsetT{0};
    const OffsetT mid_here = first + (last - first) / 2;
    const KeyT key_here    = active ? array[mid_here] : KeyT{};
    const OffsetT total_w  = mway_warp_reduce_add(mask, weight);

    OffsetT below = 0; // weight of live lanes ordered strictly before this lane's (key, lane)
    for (unsigned m = live; m != 0u; m &= m - 1u)
    {
      const int l          = __ffs(static_cast<int>(m)) - 1;
      const KeyT key_l     = mway_shfl_sync(mask, key_here, l);
      const OffsetT w_l    = mway_shfl_sync(mask, weight, l);
      const bool before_me = binary_pred(key_l, key_here) || (!binary_pred(key_here, key_l) && l < lane);
      if (active && before_me)
      {
        below += w_l;
      }
    }
    // Exactly one live lane's [below, below + weight) straddles total_w / 2.
    const bool is_median = active && (2 * below <= total_w) && (2 * (below + weight) > total_w);
    const unsigned med   = __ballot_sync(mask, is_median);
    const int leader     = __ffs(static_cast<int>(med)) - 1;
    const KeyT pivot     = mway_shfl_sync(mask, key_here, leader);
#  else
    OffsetT best_w = active ? (last - first) : OffsetT{0};
    int best_l     = lane;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = 16; offset > 0; offset >>= 1)
    {
      const OffsetT w2 = mway_shfl_xor_sync(mask, best_w, offset);
      const int l2     = __shfl_xor_sync(mask, best_l, offset);
      if (w2 > best_w || (w2 == best_w && l2 < best_l))
      {
        best_w = w2;
        best_l = l2;
      }
    }
    const int leader = best_l; // ties -> lowest lane
    OffsetT mid_here = first + (last - first) / 2;
    // Proportional pivots (Floyd-Rivest style) instead of the midpoint, for the first two rounds of a wide search. For
    // random data the target rank `remaining` among the W keys still alive sits near the proportional index first +
    // (remaining / W) * (last - first) of the leader's window. Round 1 pivots delta below it, round 2 delta above
    // (re-estimated on the windows round 1 left), delta = mway_prop_mult * sqrt(window): once both sides are cut, every
    // lane's window is the stretch between the two pivots. Any pivot is correct (the rank test still decides the side);
    // a bad estimate only costs rounds, and later rounds are the midpoint rule again. The index is snapped to an
    // absolute grid on the leader's array so neighbouring diagonals, whose estimates differ by a few elements, load the
    // same element (unsnapped, they lost their L1 sector sharing: hit rate 84-93 % -> 56-67 %).
    if (round_ix <= 2 && best_w >= static_cast<OffsetT>(mway_prop_min))
    {
      const OffsetT w_all = mway_warp_reduce_add(mask, active ? last - first : OffsetT{0});
      const float q       = static_cast<float>(remaining) / static_cast<float>(w_all);
      const float len     = static_cast<float>(last - first);
      const OffsetT delta = static_cast<OffsetT>(mway_prop_mult * sqrtf(len));
      const OffsetT cent  = first + static_cast<OffsetT>(q * len);
      OffsetT idx         = round_ix == 1 ? (cent > first + delta ? cent - delta : first) : cent + delta;
      const int lg        = 63 - __clzll(static_cast<unsigned long long>(delta > OffsetT{1} ? delta : OffsetT{1}));
      const int sh        = lg > mway_prop_qshift ? lg - mway_prop_qshift : 0;
      idx                 = mway_snap_pivot_index(idx, first, last, sh, grid_origin);
      mid_here            = idx;
    }
    const KeyT pivot_local = (lane == leader) ? array[mid_here] : KeyT{};
    const KeyT pivot       = mway_shfl_sync(mask, pivot_local, leader);
#  endif

    const OffsetT lower      = active ? mway_lower_bound<Probes>(array, first, last, pivot, binary_pred) : first;
    const OffsetT upper      = active ? mway_upper_bound_gallop(array, lower, last, pivot, binary_pred) : first;
    const OffsetT less_here  = lower - first;
    const OffsetT equal_here = upper - lower;

    const OffsetT global_less  = mway_warp_reduce_add(mask, less_here);
    const OffsetT global_equal = mway_warp_reduce_add(mask, equal_here);

    if (remaining < global_less)
    {
      last = lower;
    }
    else if (remaining > global_less + global_equal)
    {
      first = upper;
      remaining -= global_less + global_equal;
    }
    else
    {
      // The target rank falls inside the equal-to-pivot band. Resolve deterministically by lane
      // (array) order: give each array its prefix-ordered share of the needed equal-valued items.
      const OffsetT equal_needed    = remaining - global_less;
      const OffsetT equal_before_me = mway_warp_exclusive_scan_add(mask, equal_here);
      OffsetT take                  = equal_needed - equal_before_me;
      take                          = take < OffsetT{0} ? OffsetT{0} : take;
      take                          = take > equal_here ? equal_here : take;
      first                         = lower + take;
      last                          = first; // resolved: drops out of the next ballot
    }
  }
  return first;
}

// relative_span_limit is a proven geometry bound on the sum of lane widths:
// coarse_stride * merge_tile for bracketed fine search, or K*(K+1)*sample_stride
// for sampled refinement. Zero means unbounded. Global cuts and pointers stay wide.
template <int Probes, typename KeyT, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT mway_select_ranged_relative32(
  const KeyT* array,
  OffsetT first,
  OffsetT last,
  int k,
  OffsetT remaining,
  OffsetT relative_span_limit,
  BinaryPred binary_pred,
  unsigned mask = 0xffffffffu)
{
  if constexpr (sizeof(OffsetT) > sizeof(::cuda::std::int32_t))
  {
    if (relative_span_limit > 0 && relative_span_limit <= static_cast<OffsetT>(mway_relative32_max_span))
    {
      const KeyT* rebased = array != nullptr ? array + first : nullptr;
      const auto local    = mway_select_ranged<Probes>(
        rebased,
        ::cuda::std::int32_t{0},
        static_cast<::cuda::std::int32_t>(last - first),
        k,
        static_cast<::cuda::std::int32_t>(remaining),
        binary_pred,
        mask,
        first);
      return first + static_cast<OffsetT>(local);
    }
  }
  return mway_select_ranged<Probes>(array, first, last, k, remaining, binary_pred, mask);
}

template <typename KeyT, typename OffsetT, typename BinaryPred>
_CCCL_DEVICE_API _CCCL_FORCEINLINE OffsetT
mway_select(const KeyT* array, OffsetT count, int k, OffsetT diag, BinaryPred binary_pred, unsigned mask = 0xffffffffu)
{
  const int lane = static_cast<int>(threadIdx.x & 31u);
  return mway_select_ranged(array, OffsetT{0}, lane < k ? count : OffsetT{0}, k, diag, binary_pred, mask);
}
} // namespace __multiway
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILER(NVCC)

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_PARTITION_H
