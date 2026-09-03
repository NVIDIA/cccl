//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Shared-address-space sort engine for `sharded::sort` (tier 2).
 *
 * This is the places-rung engine of the two-tier sort design: where every
 * shard lives in ONE address space (locality domains of a single device, or
 * the device itself), the cross-place combine can use what this rung shares —
 * direct loads across shard boundaries — instead of the message-passing verbs
 * the ranks rung requires. The distributed (MGMN) engine remains the
 * portability path behind the same `sharded::sort` name; this engine is a
 * specialization selected when the rung allows it (see `sort_engine` in
 * `sort.cuh`).
 *
 * Structure (three phases, plain event ordering between them):
 *
 *  1. LOCAL SORT, out of place: each place sorts its shard into a per-place
 *     auxiliary run (`cub::DeviceRadixSort` for arithmetic keys under the
 *     default ascending/descending orders, `cub::DeviceMergeSort` for
 *     arbitrary comparators).
 *
 *  2. EXACT SPLITTERS by multi-sequence selection: because every sorted run
 *     is visible in the shared address space, the global splitters can be
 *     COMPUTED rather than estimated — one tiny kernel binary-searches, for
 *     each shard boundary, the element of that exact global rank across the P
 *     sorted runs (ties broken by (run, index), so the selection is total and
 *     deterministic). The selected split positions land exactly on the
 *     container's fixed shard boundaries: no sampling rounds, no histograms,
 *     no tolerance.
 *
 *  3. FUSED GATHER-MERGE: each destination place k-way-merges the P selected
 *     sub-ranges directly out of the source runs and writes straight into its
 *     own shard storage. The output lands in the original boundaries by
 *     construction, so no post-pass redistribution is needed, and the
 *     contiguous (`allocate_contiguous`) backing reads as one sorted array
 *     through the base pointer.
 *
 * Traffic: one read + one write for the local sort's logical passes plus one
 * read + one write for the gather-merge — a small constant number of passes
 * over the data, all through the shared address space.
 *
 * Determinism: the local sorts are keys-only (unique result as a multiset),
 * the selection is exact and tie-broken totally, and the merges are
 * deterministic — repeated runs are bitwise identical.
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/stream>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <functional>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
/// @brief Maximum number of places the multiselect kernel accepts by value.
inline constexpr int max_places = 64;

/// @brief The P sorted runs, passed to the selection kernel by value.
template <typename _Tp>
struct runs_desc
{
  const _Tp* data[max_places];
  size_t n[max_places];
  int p;
};

/// @brief The target global ranks (the interior shard boundaries), passed to
/// the selection kernel by value — small, so no staging buffer is needed.
struct targets_desc
{
  size_t rank[max_places - 1];
};

/**
 * @brief Multi-sequence selection: for each target global rank, find the
 * per-run split positions whose prefixes are exactly the target's prefix of
 * the merged order.
 *
 * The merged order is the total order (key, run, index): ties on the key are
 * broken by run then by index, so every element has a unique global rank and
 * the selection is deterministic. For target rank R, the element of rank R is
 * located by, per run, binary-searching the smallest index whose global rank
 * reaches R (the global rank of run r's element m with key x is
 * `m + sum_{i<r} upper_bound_i(x) + sum_{i>r} lower_bound_i(x)`); exactly one
 * run contains it. The split positions written for target t are, per run, the
 * count of elements ordered before that element; their sum is R by
 * construction.
 *
 * One thread per target; each costs O(P^2 log^2 n) — microseconds of work for
 * the place counts of this rung.
 *
 * @param[in]  runs     the P sorted runs
 * @param[in]  targets  `num_targets` global ranks (the interior shard boundaries)
 * @param[out] splits   row-major `[num_targets][P]` split positions
 */
template <typename _Tp, typename _Compare>
__global__ void
multiselect_kernel(runs_desc<_Tp> runs, targets_desc targets, int num_targets, size_t* splits, _Compare cmp)
{
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= num_targets)
  {
    return;
  }

  const size_t R = targets.rank[t];
  size_t* out    = splits + static_cast<size_t>(t) * static_cast<size_t>(runs.p);

  size_t total = 0;
  for (int i = 0; i < runs.p; i++)
  {
    total += runs.n[i];
  }

  if (R == 0 || R >= total)
  {
    for (int i = 0; i < runs.p; i++)
    {
      out[i] = (R == 0) ? 0 : runs.n[i];
    }
    return;
  }

  if (runs.p == 2)
  {
    // Two runs: the classic merge-path diagonal search — one binary search of
    // ~log2(n) steps with two loads each, instead of the general nested
    // search. Finds the same unique rank-R prefix (ties to run 0).
    const _Tp* a = runs.data[0];
    const _Tp* b = runs.data[1];
    size_t lo    = (R > runs.n[1]) ? (R - runs.n[1]) : 0;
    size_t hi    = (R < runs.n[0]) ? R : runs.n[0];
    while (lo < hi)
    {
      const size_t mid = lo + (hi - lo) / 2;
      // a[mid] belongs to the prefix iff a[mid] <= b[R-1-mid] (tie to run 0).
      if (!cmp(b[R - 1 - mid], a[mid]))
      {
        lo = mid + 1;
      }
      else
      {
        hi = mid;
      }
    }
    out[0] = lo;
    out[1] = R - lo;
    return;
  }

  // Number of elements of run i ordered before key x owned by run r.
  auto count_before = [&](int i, int r, const _Tp& x) -> size_t {
    const _Tp* b = runs.data[i];
    const _Tp* e = b + runs.n[i];
    return static_cast<size_t>(
      (i < r) ? (::cuda::std::upper_bound(b, e, x, cmp) - b) : (::cuda::std::lower_bound(b, e, x, cmp) - b));
  };

  // Global rank of run r's element m (key x) under the total order.
  auto rank_of = [&](int r, size_t m, const _Tp& x) -> size_t {
    size_t rank = m;
    for (int i = 0; i < runs.p; i++)
    {
      if (i != r)
      {
        rank += count_before(i, r, x);
      }
    }
    return rank;
  };

  for (int r = 0; r < runs.p; r++)
  {
    const size_t n_r = runs.n[r];
    if (n_r == 0)
    {
      continue;
    }

    // Smallest m with rank_of(r, m) >= R; rank_of is strictly increasing in m.
    size_t lo = 0;
    size_t hi = n_r;
    while (lo < hi)
    {
      const size_t mid = lo + (hi - lo) / 2;
      if (rank_of(r, mid, runs.data[r][mid]) >= R)
      {
        hi = mid;
      }
      else
      {
        lo = mid + 1;
      }
    }

    if (lo < n_r)
    {
      const _Tp x = runs.data[r][lo];
      if (rank_of(r, lo, x) == R)
      {
        // Found the rank-R element: split every run before it.
        for (int i = 0; i < runs.p; i++)
        {
          out[i] = (i == r) ? lo : count_before(i, r, x);
        }
        return;
      }
    }
  }

  // Unreachable: ranks are a bijection onto [0, total), so some run contains
  // rank R. The caller cross-checks split sums against the exact boundaries.
}

/// @brief Comparators under which arithmetic keys can take the radix path.
template <typename _Tp, typename _Compare>
inline constexpr bool radix_ascending =
  ::cuda::std::is_arithmetic_v<_Tp>
  && (::cuda::std::is_same_v<_Compare, ::cuda::std::less<_Tp>> || ::cuda::std::is_same_v<_Compare, ::cuda::std::less<>>
      || ::cuda::std::is_same_v<_Compare, ::std::less<_Tp>>);

template <typename _Tp, typename _Compare>
inline constexpr bool radix_descending =
  ::cuda::std::is_arithmetic_v<_Tp>
  && (::cuda::std::is_same_v<_Compare, ::cuda::std::greater<_Tp>>
      || ::cuda::std::is_same_v<_Compare, ::cuda::std::greater<>>
      || ::cuda::std::is_same_v<_Compare, ::std::greater<_Tp>>);

/// @brief A stream-ordered allocation to release once the sort has drained.
struct scoped_alloc
{
  places::place_memory_resource mr;
  void* ptr;
  size_t bytes;
  cudaStream_t stream;
};

/// @brief One sorted sub-range feeding a destination's merge.
template <typename _Tp>
struct merge_range
{
  const _Tp* ptr;
  size_t count;
};

/**
 * @brief Merge @p ranges into `dest[0, total)` on @p stream, deterministically.
 *
 * 1 range is a copy and 2 ranges a single `DeviceMerge`; more ranges run a
 * balanced pairwise merge tree ping-ponging between the destination and one
 * scratch buffer (allocated here from the destination's place and registered
 * for deferred release).
 */
template <typename _Tp, typename _Compare, typename _Env>
void merge_into(
  const ::std::vector<merge_range<_Tp>>& ranges,
  _Tp* dest,
  size_t total,
  const data_place& dplace,
  cudaStream_t stream,
  const _Env& env,
  _Compare cmp,
  ::std::vector<scoped_alloc>* deferred)
{
  if (ranges.empty() || total == 0)
  {
    return;
  }

  if (ranges.size() == 1)
  {
    cuda_safe_call(cudaMemcpyAsync(dest, ranges[0].ptr, total * sizeof(_Tp), cudaMemcpyDefault, stream));
    return;
  }

  if (ranges.size() == 2)
  {
    cuda_safe_call(cub::DeviceMerge::MergeKeys(
      ranges[0].ptr,
      static_cast<::cuda::std::int64_t>(ranges[0].count),
      ranges[1].ptr,
      static_cast<::cuda::std::int64_t>(ranges[1].count),
      dest,
      cmp,
      env));
    return;
  }

  // Pairwise merge tree. Levels alternate between the destination and one
  // scratch buffer; a trailing odd node is copied so every level's inputs
  // live outside its output buffer.
  places::place_memory_resource mr(dplace);
  _Tp* scratch = static_cast<_Tp*>(mr.allocate(::cuda::stream_ref{stream}, total * sizeof(_Tp), alignof(_Tp)));
  deferred->push_back(scoped_alloc{mr, scratch, total * sizeof(_Tp), stream});

  ::std::vector<merge_range<_Tp>> cur = ranges;
  ::std::vector<merge_range<_Tp>> next;
  _Tp* bufs[2] = {dest, scratch};
  int wb       = 0;

  while (cur.size() > 1)
  {
    next.clear();
    size_t off = 0;
    size_t i   = 0;
    for (; i + 1 < cur.size(); i += 2)
    {
      cuda_safe_call(cub::DeviceMerge::MergeKeys(
        cur[i].ptr,
        static_cast<::cuda::std::int64_t>(cur[i].count),
        cur[i + 1].ptr,
        static_cast<::cuda::std::int64_t>(cur[i + 1].count),
        bufs[wb] + off,
        cmp,
        env));
      const size_t merged = cur[i].count + cur[i + 1].count;
      next.push_back(merge_range<_Tp>{bufs[wb] + off, merged});
      off += merged;
    }
    if (i < cur.size())
    {
      cuda_safe_call(
        cudaMemcpyAsync(bufs[wb] + off, cur[i].ptr, cur[i].count * sizeof(_Tp), cudaMemcpyDefault, stream));
      next.push_back(merge_range<_Tp>{bufs[wb] + off, cur[i].count});
    }
    cur.swap(next);
    wb ^= 1;
  }

  if (cur[0].ptr != dest)
  {
    cuda_safe_call(cudaMemcpyAsync(dest, cur[0].ptr, total * sizeof(_Tp), cudaMemcpyDefault, stream));
  }
}

/**
 * @brief True when every shard lives in one shared address space this engine
 * may load across directly: all shards on device-backed places (device,
 * locality domain, green context) of the SAME underlying device.
 */
template <typename _Tp>
[[nodiscard]] bool one_shared_address_space(const sharded_array<_Tp>& data)
{
  if (data.num_shards() == 0)
  {
    return false;
  }
  int dev = -1;
  for (size_t g = 0; g < data.num_shards(); g++)
  {
    const int d = places::device_ordinal(data.shard(g).place);
    if (d < 0)
    {
      return false; // host / managed / composite: not this engine's rung
    }
    if (dev < 0)
    {
      dev = d;
    }
    else if (d != dev)
    {
      return false;
    }
  }
  return true;
}

/**
 * @brief The shared-address-space sort: local out-of-place sorts, exact
 * splitters by multi-sequence selection, fused gather-merge into the shards'
 * own storage. See the file-level comment.
 */
template <typename _Tp, typename _Compare>
void sort_shared_va(place_group& group, sharded_array<_Tp>& data, _Compare comp)
{
  const size_t p = data.num_shards();
  _CCCL_ASSERT(p >= 1, "engine invoked on an empty array");
  if (p > static_cast<size_t>(max_places))
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::sort (shared_va): more places (" + ::std::to_string(p) + ") than the engine supports ("
                  + ::std::to_string(max_places) + ")");
  }

  ::std::vector<scoped_alloc> deferred; // released after the final drain
  ::std::vector<_Tp*> sorted(p, nullptr); // per-place sorted runs (aux)

  // --------------------------------------------------------------------
  // Phase 1: local sorts, out of place, on each shard's stream.
  // --------------------------------------------------------------------
  data.each_shard->*[&](size_t g, auto& s) {
    places::place_memory_resource mr(s.place);
    _Tp* aux = static_cast<_Tp*>(mr.allocate(::cuda::stream_ref{s.stream}, s.size * sizeof(_Tp), alignof(_Tp)));
    deferred.push_back(scoped_alloc{mr, aux, s.size * sizeof(_Tp), s.stream});
    sorted[g] = aux;

    const auto env = place_group::env(s.place, s.stream);
    if constexpr (radix_ascending<_Tp, _Compare>)
    {
      cuda_safe_call(cub::DeviceRadixSort::SortKeys(s.data, aux, s.size, 0, static_cast<int>(sizeof(_Tp) * 8), env));
    }
    else if constexpr (radix_descending<_Tp, _Compare>)
    {
      cuda_safe_call(
        cub::DeviceRadixSort::SortKeysDescending(s.data, aux, s.size, 0, static_cast<int>(sizeof(_Tp) * 8), env));
    }
    else
    {
      cuda_safe_call(cub::DeviceMergeSort::SortKeysCopy(s.data, aux, s.size, comp, env));
    }
  };

  // --------------------------------------------------------------------
  // Phase 2: exact splitters at the container's fixed shard boundaries.
  // --------------------------------------------------------------------
  ::std::vector<size_t> h_begin(p * p, 0); // row-major [dest][run] range begins
  ::std::vector<size_t> h_end(p * p, 0);

  if (p > 1)
  {
    runs_desc<_Tp> runs{};
    runs.p = static_cast<int>(p);
    for (size_t g = 0; g < p; g++)
    {
      runs.data[g] = sorted[g];
      runs.n[g]    = data.shard(g).size;
    }

    const int num_targets = static_cast<int>(p - 1);

    // The targets travel by kernel parameter; the splits come back through a
    // small pooled device buffer and one async copy — no per-call pinned
    // allocation on the critical path.
    targets_desc targets{};
    for (size_t k = 1; k < p; k++)
    {
      targets.rank[k - 1] = data.shard(k).global_offset; // exact fixed boundary
    }

    const size_t splits_count = static_cast<size_t>(num_targets) * p;
    ::std::vector<size_t> h_splits(splits_count);

    // The selection runs on shard 0's place, after every local sort.
    const auto& s0 = data.shard(0);
    {
      exec_place_scope scope(s0.exec);
      places::place_memory_resource mr(s0.place);
      auto* d_splits = static_cast<size_t*>(
        mr.allocate(::cuda::stream_ref{s0.stream}, splits_count * sizeof(size_t), alignof(size_t)));

      for (size_t g = 1; g < p; g++)
      {
        ::cuda::stream_ref{s0.stream}.wait(::cuda::stream_ref{data.shard(g).stream});
      }
      multiselect_kernel<_Tp, _Compare>
        <<<1, static_cast<unsigned>(num_targets), 0, s0.stream>>>(runs, targets, num_targets, d_splits, comp);
      cuda_safe_call(cudaGetLastError());
      cuda_safe_call(
        cudaMemcpyAsync(h_splits.data(), d_splits, splits_count * sizeof(size_t), cudaMemcpyDeviceToHost, s0.stream));
      cuda_safe_call(cudaStreamSynchronize(s0.stream));
      mr.deallocate(::cuda::stream_ref{s0.stream}, d_splits, splits_count * sizeof(size_t), alignof(size_t));
    }
    // Host now sees the splits, and every local sort has drained (the kernel
    // waited on all shard streams and we synced its stream).

    for (size_t k = 0; k < p; k++)
    {
      for (size_t i = 0; i < p; i++)
      {
        h_begin[k * p + i] = (k == 0) ? 0 : h_splits[(k - 1) * p + i];
        h_end[k * p + i]   = (k == p - 1) ? data.shard(i).size : h_splits[k * p + i];
      }
    }
  }
  else
  {
    // Single place: the whole run is destination 0's single range.
    h_begin[0] = 0;
    h_end[0]   = data.shard(0).size;
    // The gather below runs on the same stream as the sort: in order.
  }

  // --------------------------------------------------------------------
  // Phase 3: fused gather-merge into each shard's own storage.
  // --------------------------------------------------------------------
  data.each_shard->*[&](size_t k, auto& s) {
    ::std::vector<merge_range<_Tp>> ranges;
    size_t total = 0;
    for (size_t i = 0; i < p; i++)
    {
      const size_t b = h_begin[k * p + i];
      const size_t e = h_end[k * p + i];
      _CCCL_ASSERT(b <= e && e <= data.shard(i).size, "selection produced an invalid range");
      if (e > b)
      {
        ranges.push_back(merge_range<_Tp>{sorted[i] + b, e - b});
        total += e - b;
      }
    }

    // The exact selection lands exactly on the fixed boundary: the merged
    // total IS the shard's original size. This is the engine's boundary
    // contract; a violation would corrupt neighboring shards, so verify.
    _CCCL_VERIFY(total == s.size, "shared_va sort: exact splits must land on the shard boundaries");

    const auto env = place_group::env(s.place, s.stream);
    merge_into(ranges, s.data, total, s.place, s.stream, env, comp, &deferred);
  };

  // --------------------------------------------------------------------
  // Synchronous contract, then release the stream-ordered temporaries.
  // --------------------------------------------------------------------
  data.sync();
  for (auto& a : deferred)
  {
    a.mr.deallocate(::cuda::stream_ref{a.stream}, a.ptr, a.bytes, alignof(_Tp));
  }
  (void) group;
}
} // namespace reserved
} // namespace cuda::experimental::sharded
