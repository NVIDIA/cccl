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
 *     COMPUTED rather than estimated — for each shard boundary, the element
 *     of that exact global rank across the P sorted runs (ties broken by
 *     (run, index), so the selection is total and deterministic). The
 *     selected split positions land exactly on the container's fixed shard
 *     boundaries: no sampling rounds, no histograms, no tolerance.
 *     P == 2 is one merge-path diagonal search. P > 2 is two exact
 *     selections on LOCAL copies: first over a strided sample of every run
 *     (bulk-gathered to the selecting place), which brackets each run's split
 *     within a window of (P+2)*stride elements; then over those windows
 *     (bulk-copied local). Both selections are one thread per element, each
 *     computing its own rank with P independent binary searches — no serial
 *     chain of dependent loads, and no dependent load ever crosses the
 *     fabric. (The previous nested search was O(P^2 log^2 n) dependent loads
 *     on one thread per boundary: milliseconds over NVLink at P = 4.)
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
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
/// @brief Maximum number of places the selection stage supports.
inline constexpr int max_places = 64;

/// @brief The P sorted runs, passed to the selection kernels by value.
template <typename _Tp>
struct runs_desc
{
  const _Tp* data[max_places];
  size_t n[max_places];
  int p;
};

/// @brief Number of elements of run @p i ordered before key @p x owned by run
/// @p r under the (key, run, index) total order: ties go to the lower run.
template <typename _Tp, typename _Compare>
__device__ inline size_t count_before(const _Tp* b, size_t n, int i, int r, const _Tp& x, _Compare cmp)
{
  const _Tp* e = b + n;
  return static_cast<size_t>(
    (i < r) ? (::cuda::std::upper_bound(b, e, x, cmp) - b) : (::cuda::std::lower_bound(b, e, x, cmp) - b));
}

/**
 * @brief Two-run selection: the classic merge-path diagonal search — one
 * binary search of ~log2(n) steps with two loads each. Finds the unique
 * rank-R prefix (ties to run 0). One thread per target.
 */
template <typename _Tp, typename _Compare>
__global__ void merge_path_select_kernel(
  const _Tp* a, size_t na, const _Tp* b, size_t nb, const size_t* ranks, int num_targets, size_t* splits, _Compare cmp)
{
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= num_targets)
  {
    return;
  }
  const size_t R = ranks[t];
  size_t lo      = (R > nb) ? (R - nb) : 0;
  size_t hi      = (R < na) ? R : na;
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
  splits[2 * t]     = lo;
  splits[2 * t + 1] = R - lo;
}

/**
 * @brief One exact selection problem for `rank_select_kernel`: P sorted
 * arrays (all local to the selecting device) and a sorted list of target
 * ranks. For each target the kernel finds the element of that rank in the
 * (key, array, index) total order and writes, per array, the count of
 * elements ordered before it (+ `base`, when given).
 */
template <typename _Tp>
struct select_problem
{
  const _Tp* arr[max_places];
  size_t n[max_places];
  int p;
  int num_targets;
  const size_t* ranks; //!< [num_targets], non-decreasing
  const size_t* base; //!< [num_targets * p] added to the output, or nullptr
  size_t* out; //!< [num_targets * p]
};

/**
 * @brief Exact multi-sequence selection, one thread per element: each
 * element computes its own global rank with P independent binary searches
 * (no dependent chain across elements), and the element whose rank matches a
 * target writes that target's splits. Ranks are a bijection onto
 * [0, total), so exactly one element answers each target.
 *
 * All arrays are expected in the selecting device's own memory: the
 * searches are latency-bound, and the caller has already turned every remote
 * access into a bulk copy (samples, then windows).
 *
 * @param problems  one problem per `blockIdx.y`
 */
template <typename _Tp, typename _Compare>
__global__ void rank_select_kernel(const select_problem<_Tp>* problems, _Compare cmp)
{
  const select_problem<_Tp>& pb = problems[blockIdx.y];
  const int p                   = pb.p;

  size_t total = 0;
  for (int i = 0; i < p; i++)
  {
    total += pb.n[i];
  }

  for (size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < total;
       e += static_cast<size_t>(gridDim.x) * blockDim.x)
  {
    // Locate (array r, index m) of element e in the concatenation.
    int r    = 0;
    size_t m = e;
    while (m >= pb.n[r])
    {
      m -= pb.n[r];
      r++;
    }
    const _Tp x = pb.arr[r][m];

    size_t rank = m;
    for (int i = 0; i < p; i++)
    {
      if (i != r)
      {
        rank += count_before(pb.arr[i], pb.n[i], i, r, x, cmp);
      }
    }

    // Targets are sorted (duplicates allowed when several boundaries collapse
    // onto one sample): answer every target of this rank.
    const size_t* t_end = pb.ranks + pb.num_targets;
    const size_t* t     = ::cuda::std::lower_bound(pb.ranks, t_end, rank);
    for (; t != t_end && *t == rank; ++t)
    {
      const int ti = static_cast<int>(t - pb.ranks);
      for (int i = 0; i < p; i++)
      {
        size_t c = (i == r) ? m : count_before(pb.arr[i], pb.n[i], i, r, x, cmp);
        if (pb.base != nullptr)
        {
          c += pb.base[static_cast<size_t>(ti) * p + i];
        }
        pb.out[static_cast<size_t>(ti) * p + i] = c;
      }
    }
  }
}

/// @brief Where each run's samples live in the concatenated sample buffer.
struct sample_layout
{
  size_t off[max_places + 1];
};

/// @brief samples[off[i] + j] = runs.data[i][j * stride]: a scattered but
/// fully parallel gather (remote loads are fine here, nothing depends on them).
template <typename _Tp>
__global__ void gather_samples_kernel(runs_desc<_Tp> runs, sample_layout lay, size_t stride, _Tp* samples)
{
  const size_t idx   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = lay.off[runs.p];
  if (idx >= total)
  {
    return;
  }
  int r = 0;
  while (idx >= lay.off[r + 1])
  {
    r++;
  }
  const size_t j = idx - lay.off[r];
  samples[idx]   = runs.data[r][j * stride];
}

/**
 * @brief From the sample-level answer for target t (`c[t][i]` = samples of
 * run i ordered before the chosen sample x0), derive the window of run i that
 * is guaranteed to contain the exact split, and describe the window-level
 * selection problem. One thread per (target, run).
 *
 * With x0 the element of sample-rank floor(R / stride), x0 is in the rank-R
 * prefix and at most (P+1)*stride ranks before its end; the full-run count
 * before x0 is itself bracketed by the samples within `stride`. Hence
 *   split_i in [ (c_i-1)*stride + 1 , c_i*stride + (P+1)*stride ]   (c_i >= 1)
 *   split_i in [ 0 , (P+1)*stride ]                                  (c_i == 0)
 * clamped to [0, n_i]. Window width <= (P+2)*stride = `win_cap`.
 */
template <typename _Tp>
__global__ void build_windows_kernel(
  runs_desc<_Tp> runs,
  const size_t* c,
  size_t stride,
  size_t win_cap,
  const size_t* target_ranks,
  int num_targets,
  _Tp* winbuf,
  size_t* w_lo,
  size_t* w_rank,
  select_problem<_Tp>* problems,
  size_t* out_splits)
{
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int p   = runs.p;
  if (idx >= num_targets * p)
  {
    return;
  }
  const int t = idx / p;
  const int i = idx % p;

  auto window = [&](int ii, size_t& lo, size_t& hi) {
    const size_t ci = c[static_cast<size_t>(t) * p + ii];
    lo              = (ci == 0) ? 0 : (ci - 1) * stride + 1;
    hi              = ci * stride + (static_cast<size_t>(p) + 1) * stride;
    if (hi > runs.n[ii])
    {
      hi = runs.n[ii];
    }
    if (lo > hi)
    {
      lo = hi;
    }
  };

  size_t lo, hi;
  window(i, lo, hi);
  w_lo[idx]          = lo;
  problems[t].arr[i] = winbuf + static_cast<size_t>(idx) * win_cap;
  problems[t].n[i]   = hi - lo;

  if (i == 0)
  {
    size_t sum_lo = 0;
    for (int ii = 0; ii < p; ii++)
    {
      size_t l, h;
      window(ii, l, h);
      sum_lo += l;
    }
    w_rank[t]               = target_ranks[t] - sum_lo; // elements before the windows are all in the prefix
    problems[t].p           = p;
    problems[t].num_targets = 1;
    problems[t].ranks       = w_rank + t;
    problems[t].base        = w_lo + static_cast<size_t>(t) * p;
    problems[t].out         = out_splits + static_cast<size_t>(t) * p;
  }
}

/// @brief Copy every window into the local window buffer: contiguous, so the
/// remote reads coalesce. One thread per slot.
template <typename _Tp>
__global__ void copy_windows_kernel(
  runs_desc<_Tp> runs,
  size_t win_cap,
  int num_targets,
  const size_t* w_lo,
  const select_problem<_Tp>* problems,
  _Tp* winbuf)
{
  const size_t idx   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t slots = static_cast<size_t>(num_targets) * runs.p * win_cap;
  if (idx >= slots)
  {
    return;
  }
  const size_t ti = idx / win_cap; // (target, run) flattened
  const size_t k  = idx % win_cap;
  const int t     = static_cast<int>(ti / runs.p);
  const int i     = static_cast<int>(ti % runs.p);
  if (k < problems[t].n[i])
  {
    winbuf[idx] = runs.data[i][w_lo[ti] + k];
  }
}

/// @brief Integer ceil division.
__host__ __device__ inline size_t ceil_div(size_t a, size_t b)
{
  return (a + b - 1) / b;
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
  // live outside its output buffer. The first level's buffer is chosen so
  // that the LAST level lands in the destination: the level that reads the
  // (possibly remote) runs is the expensive one, the local levels are cheap,
  // and a trailing whole-shard copy would be pure waste.
  places::place_memory_resource mr(dplace);
  _Tp* scratch = static_cast<_Tp*>(mr.allocate(::cuda::stream_ref{stream}, total * sizeof(_Tp), alignof(_Tp)));
  deferred->push_back(scoped_alloc{mr, scratch, total * sizeof(_Tp), stream});

  ::std::vector<merge_range<_Tp>> cur = ranges;
  ::std::vector<merge_range<_Tp>> next;
  _Tp* bufs[2] = {dest, scratch};
  int levels   = 0;
  for (size_t n = cur.size(); n > 1; n = (n + 1) / 2)
  {
    levels++;
  }
  int wb = (levels - 1) & 1; // level l writes bufs[(wb + l) & 1]; level levels-1 must write bufs[0]

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

  _CCCL_ASSERT(cur[0].ptr == dest, "merge tree parity: the last level must land in the destination");
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
    runs.p       = static_cast<int>(p);
    size_t total = 0;
    for (size_t g = 0; g < p; g++)
    {
      runs.data[g] = sorted[g];
      runs.n[g]    = data.shard(g).size;
      total += runs.n[g];
    }

    const size_t num_targets = p - 1;
    ::std::vector<size_t> h_splits(num_targets * p);

    // Trivial targets (empty leading/trailing shards) are answered on the
    // host; the kernels see only 0 < R < total, in non-decreasing order.
    ::std::vector<size_t> h_ranks; // active targets' ranks
    ::std::vector<size_t> h_active; // and their indices k-1
    for (size_t k = 1; k < p; k++)
    {
      const size_t R = data.shard(k).global_offset; // exact fixed boundary
      if (R == 0 || R >= total)
      {
        for (size_t i = 0; i < p; i++)
        {
          h_splits[(k - 1) * p + i] = (R == 0) ? 0 : runs.n[i];
        }
      }
      else
      {
        h_ranks.push_back(R);
        h_active.push_back(k - 1);
      }
    }
    const size_t T = h_ranks.size();

    if (T > 0)
    {
      // The selection runs on shard 0's place, after every local sort. Every
      // remote access below is a bulk copy (samples, then windows); the
      // latency-bound searches only ever touch local memory.
      const auto& s0 = data.shard(0);
      exec_place_scope scope(s0.exec);
      places::place_memory_resource mr(s0.place);
      const ::cuda::stream_ref st{s0.stream};
      for (size_t g = 1; g < p; g++)
      {
        st.wait(::cuda::stream_ref{data.shard(g).stream});
      }

      ::std::vector<scoped_alloc> tmp; // released after the host sync below
      auto dalloc = [&](size_t bytes) -> void* {
        bytes    = bytes == 0 ? 1 : bytes;
        void* pt = mr.allocate(st, bytes, 256);
        tmp.push_back(scoped_alloc{mr, pt, bytes, s0.stream});
        return pt;
      };
      auto h2d = [&](void* dst, const void* src, size_t bytes) {
        cuda_safe_call(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s0.stream));
      };

      auto* d_ranks  = static_cast<size_t*>(dalloc(T * sizeof(size_t)));
      auto* d_splits = static_cast<size_t*>(dalloc(T * p * sizeof(size_t)));
      h2d(d_ranks, h_ranks.data(), T * sizeof(size_t));

      if (p == 2)
      {
        const unsigned blk = 32;
        merge_path_select_kernel<_Tp, _Compare><<<(static_cast<unsigned>(T) + blk - 1) / blk, blk, 0, s0.stream>>>(
          runs.data[0], runs.n[0], runs.data[1], runs.n[1], d_ranks, static_cast<int>(T), d_splits, comp);
        cuda_safe_call(cudaGetLastError());
      }
      else
      {
        // Sample stride: balance the sample count (N / stride) against the
        // window volume (T * P * (P+2) * stride) — equal at their geometric
        // mean, rounded to a power of two.
        size_t n_max = 0;
        for (size_t g = 0; g < p; g++)
        {
          n_max = runs.n[g] > n_max ? runs.n[g] : n_max;
        }
        size_t stride = 1;
        {
          const double win_per_stride = static_cast<double>(T) * static_cast<double>(p) * static_cast<double>(p + 2);
          const double s_opt          = ::std::sqrt(static_cast<double>(total) / win_per_stride);
          while (stride * 2 <= static_cast<size_t>(s_opt))
          {
            stride *= 2;
          }
          if (stride > n_max)
          {
            stride = n_max;
          }
        }
        const size_t win_cap = (p + 2) * stride;

        // ---- stage A: samples, gathered local ---------------------------
        sample_layout lay{};
        lay.off[0] = 0;
        for (size_t g = 0; g < p; g++)
        {
          lay.off[g + 1] = lay.off[g] + ceil_div(runs.n[g], stride);
        }
        const size_t num_samples = lay.off[p];

        auto* d_samples = static_cast<_Tp*>(dalloc(num_samples * sizeof(_Tp)));
        auto* d_sranks  = static_cast<size_t*>(dalloc(T * sizeof(size_t)));
        auto* d_c       = static_cast<size_t*>(dalloc(T * p * sizeof(size_t)));
        auto* d_prob_s  = static_cast<select_problem<_Tp>*>(dalloc(sizeof(select_problem<_Tp>)));

        ::std::vector<size_t> h_sranks(T);
        for (size_t t = 0; t < T; t++)
        {
          h_sranks[t] = h_ranks[t] / stride; // floor: x0 lands inside the prefix
        }
        select_problem<_Tp> h_prob_s{};
        for (size_t g = 0; g < p; g++)
        {
          h_prob_s.arr[g] = d_samples + lay.off[g];
          h_prob_s.n[g]   = lay.off[g + 1] - lay.off[g];
        }
        h_prob_s.p           = static_cast<int>(p);
        h_prob_s.num_targets = static_cast<int>(T);
        h_prob_s.ranks       = d_sranks;
        h_prob_s.base        = nullptr;
        h_prob_s.out         = d_c;
        h2d(d_sranks, h_sranks.data(), T * sizeof(size_t));
        h2d(d_prob_s, &h_prob_s, sizeof(h_prob_s));

        const unsigned blk = 256;
        auto blocks        = [&](size_t items) {
          return static_cast<unsigned>(ceil_div(items == 0 ? 1 : items, blk));
        };
        auto grid_x = [&](size_t items) {
          const unsigned b = blocks(items);
          return b < 4096u ? b : 4096u; // grid-stride inside the kernel
        };

        gather_samples_kernel<_Tp><<<blocks(num_samples), blk, 0, s0.stream>>>(runs, lay, stride, d_samples);
        cuda_safe_call(cudaGetLastError());
        rank_select_kernel<_Tp, _Compare><<<dim3(grid_x(num_samples), 1), blk, 0, s0.stream>>>(d_prob_s, comp);
        cuda_safe_call(cudaGetLastError());

        // ---- stage B: windows around the samples' answer, copied local ----
        auto* d_winbuf = static_cast<_Tp*>(dalloc(T * p * win_cap * sizeof(_Tp)));
        auto* d_wlo    = static_cast<size_t*>(dalloc(T * p * sizeof(size_t)));
        auto* d_wrank  = static_cast<size_t*>(dalloc(T * sizeof(size_t)));
        auto* d_prob_w = static_cast<select_problem<_Tp>*>(dalloc(T * sizeof(select_problem<_Tp>)));

        build_windows_kernel<_Tp><<<blocks(T * p), blk, 0, s0.stream>>>(
          runs, d_c, stride, win_cap, d_ranks, static_cast<int>(T), d_winbuf, d_wlo, d_wrank, d_prob_w, d_splits);
        cuda_safe_call(cudaGetLastError());
        copy_windows_kernel<_Tp><<<blocks(T * p * win_cap), blk, 0, s0.stream>>>(
          runs, win_cap, static_cast<int>(T), d_wlo, d_prob_w, d_winbuf);
        cuda_safe_call(cudaGetLastError());
        rank_select_kernel<_Tp, _Compare>
          <<<dim3(grid_x(p * win_cap), static_cast<unsigned>(T)), blk, 0, s0.stream>>>(d_prob_w, comp);
        cuda_safe_call(cudaGetLastError());
      }

      ::std::vector<size_t> h_active_splits(T * p);
      cuda_safe_call(
        cudaMemcpyAsync(h_active_splits.data(), d_splits, T * p * sizeof(size_t), cudaMemcpyDeviceToHost, s0.stream));
      cuda_safe_call(cudaStreamSynchronize(s0.stream));
      for (auto& a : tmp)
      {
        a.mr.deallocate(st, a.ptr, a.bytes, 256);
      }
      for (size_t t = 0; t < T; t++)
      {
        for (size_t i = 0; i < p; i++)
        {
          h_splits[h_active[t] * p + i] = h_active_splits[t * p + i];
        }
      }
    }
    else
    {
      // Every boundary was trivial; the local sorts still have to drain
      // before the gather-merge reads across shards.
      const ::cuda::stream_ref st{data.shard(0).stream};
      for (size_t g = 1; g < p; g++)
      {
        st.wait(::cuda::stream_ref{data.shard(g).stream});
      }
      cuda_safe_call(cudaStreamSynchronize(data.shard(0).stream));
    }
    // Host now sees the splits, and every local sort has drained.

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
