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
 * @brief `sharded_csr<T>`: a row-partitioned CSR sparse matrix, one shard per
 *        place of a `place_group`.
 *
 * Each shard is a self-contained CSR matrix for a contiguous row range:
 *
 *  - `values` / `colinds`: the shard's nnz slice of the parent arrays
 *  - `offsets`: the shard's rows+1 offsets, REBASED to start at 0
 *  - storage lives in the shard's place (for locality-domain places, the
 *    domain's localized memory; confined consumers then read domain-local
 *    bytes)
 *
 * Because each shard is a complete CSR operator over its row range, a plain
 * per-place library call (one vendor call per shard, on the shard's stream)
 * consumes it without any library changes — that is the point: the container
 * carries the placement so that CLOSED libraries, which only understand
 * pointers and a stream, can be run placement-localized. The vendor calls
 * themselves live in the opt-in `__sharded/sparse.cuh` header; this container
 * never includes vendor headers and owns NO library state: handles and plans
 * are EXPLICIT caller-held objects of that header (`cusparse_handles`,
 * `spmv_plan` / `spmm_plan`) whose lifetimes the caller can read off the
 * page — the same discipline as the cuRAND tier.
 *
 * TWO ADOPTION GRADES for matrices that already live on the device:
 *  - `adopt` (zero-copy): colinds/values shards ALIAS the caller's arrays at
 *    row-aligned nnz slices; only the rebased per-shard offsets are
 *    container-owned. The caller's arrays remain the storage (in-place value
 *    mutation IS mutation of the matrix; the arrays must outlive the
 *    container) and keep whatever placement the caller's allocator gave
 *    them — the split/confinement structure without per-place placement.
 *  - `from_device` (placed copy): container-owned per-place copies; the
 *    caller's arrays may be freed on return, and each shard's bytes live in
 *    its place — the grade that buys the measured locality win, for the
 *    price of a one-time copy at construction.
 *
 * Split points are caller-suppliable row boundaries; the default is an
 * nnz-balanced split. CAVEAT: nnz balance is not time balance — under SM
 * confinement the split finishes at max(shard time), and a skewed row-length
 * distribution can make an nnz-balanced split strongly time-imbalanced. When
 * the matrix is reused across many calls, measure per-shard solo times once
 * (e.g. `spmv_shard_times` / `spmm_shard_times` from the sparse header) and
 * rebalance with `time_balanced_boundaries`.
 *
 * The dense operands of the sparse products are NOT containers: a row
 * partition needs the whole dense operand at every place. Coherent per-place
 * replication of such operands is the binding tier's job (STF
 * `logical_data`); the container composes with it rather than absorbing it.
 *
 * VALUES ARE ORDINARY MUTABLE DATA. The values (and colinds) shards are exact
 * nnz-slices of the parent arrays, so with `contiguous = true` they are
 * backed by ONE contiguous VA range (`sharded_array::allocate_contiguous`):
 * `contiguous_values()` hands the whole values array to unmodified writers
 * (e.g. a problem-scaling transform kernel) as one normal pointer while each
 * slice keeps per-place physical placement. In-place value mutation never
 * invalidates per-shard library plans (fixed addresses; only structure is
 * frozen by a plan), whichever backing or adoption grade is used.
 *
 * Matrices whose arrays already live on the device are adopted with
 * `from_device` (offsets make one small round trip to the host for the split
 * and the per-shard rebasing; colinds/values are sliced device-to-device).
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/stream>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/fork_join.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief One row-range shard of a `sharded_csr`: a complete CSR matrix for
 *        rows [row_begin, row_begin + rows), with rebased offsets, plus its
 *        placement.
 */
template <typename _Tp>
struct csr_shard
{
  ::std::int64_t row_begin = 0; //!< first row of this shard in the parent matrix
  ::std::int64_t rows      = 0; //!< number of rows
  ::std::int64_t nnz_begin = 0; //!< first nonzero (parent indexing)
  ::std::int64_t nnz       = 0; //!< number of nonzeros
  int* offsets             = nullptr; //!< rows+1 entries, rebased (offsets[0] == 0)
  int* colinds             = nullptr; //!< nnz entries (parent column indices)
  _Tp* values              = nullptr; //!< nnz entries
  data_place place; //!< where the shard's arrays live
  exec_place exec; //!< execution place for consumers of this shard
  cudaStream_t stream = nullptr; //!< reference stream (group stream for the place)
};

/**
 * @brief A row-partitioned CSR matrix: one shard per place of a
 *        `place_group`, each shard stored in its place's memory.
 *
 * See the file-level comment for the design. The container owns the backing
 * `sharded_array`s (always the rebased offsets; colinds/values only in the
 * owning grades) and hands out per-shard views. It owns no library state.
 */
template <typename _Tp>
class sharded_csr
{
public:
  using value_type = _Tp;
  using shard_type = csr_shard<_Tp>;

  /**
   * @brief Build a row-partitioned CSR from host CSR data, one shard per
   *        place of the group, each shard stored in its place's memory.
   *
   * SYNCHRONOUS: returns with all shards populated.
   *
   * @param group Place group whose places define the partition (and provide
   *              the reference streams)
   * @param num_rows Number of rows of the matrix
   * @param num_cols Number of columns of the matrix
   * @param h_offsets Host CSR row offsets (num_rows+1 ints)
   * @param h_colinds Host CSR column indices (nnz ints)
   * @param h_values Host CSR values (nnz)
   * @param row_boundaries Optional interior split rows (ascending, size
   *              group.size()-1); shard d covers [boundary[d-1], boundary[d]).
   *              Empty = nnz-balanced split (see the time-balance caveat in
   *              the file-level comment).
   */
  sharded_csr(place_group& group,
              ::std::int64_t num_rows,
              ::std::int64_t num_cols,
              const int* h_offsets,
              const int* h_colinds,
              const _Tp* h_values,
              ::std::vector<::std::int64_t> row_boundaries = {},
              bool contiguous                              = false)
      : rows_(num_rows)
      , cols_(num_cols)
      , nnz_(h_offsets[num_rows])
  {
    init(group, h_offsets, h_colinds, h_values, mv(row_boundaries), contiguous);
  }

  /**
   * @brief Build a row-partitioned CSR from DEVICE CSR arrays.
   *
   * Same contract as the host constructor; the offsets make one small host
   * round trip (num_rows+1 ints — the split and the per-shard rebasing are
   * host-side either way) and the colinds/values slices are copied
   * device-to-device into the shards' places. SYNCHRONOUS.
   *
   * Ownership/aliasing contract (explicit): NOTHING in the returned matrix
   * aliases the caller's arrays. Per-shard offsets are rebuilt (rebased so
   * `offsets[0] == 0` in every shard) into container-owned storage, and the
   * colinds/values slices are container-owned per-place copies, so the
   * caller's `d_offsets`/`d_colinds`/`d_values` may be freed as soon as this
   * call returns. The copy is also a snapshot: later in-place writes to the
   * caller's arrays do NOT propagate to the shards — mutate the shard views
   * (or `contiguous_values()`) instead, which never invalidates the
   * per-shard library plans.
   */
  static sharded_csr from_device(
    place_group& group,
    ::std::int64_t num_rows,
    ::std::int64_t num_cols,
    const int* d_offsets,
    const int* d_colinds,
    const _Tp* d_values,
    ::std::vector<::std::int64_t> row_boundaries = {},
    bool contiguous                              = false)
  {
    ::std::vector<int> h_offsets(static_cast<size_t>(num_rows) + 1);
    cuda_safe_call(cudaMemcpy(h_offsets.data(), d_offsets, h_offsets.size() * sizeof(int), cudaMemcpyDefault));
    return sharded_csr(group, num_rows, num_cols, h_offsets.data(), d_colinds, d_values, mv(row_boundaries), contiguous);
  }

  /**
   * @brief ZERO-COPY adoption of DEVICE CSR arrays: colinds/values shards
   *        alias the caller's arrays at row-aligned nnz slices; only the
   *        rebased per-shard offsets are container-owned (built through one
   *        small host round trip of the offsets).
   *
   * Ownership/aliasing contract (the inverse of `from_device`): the caller's
   * `d_colinds`/`d_values` REMAIN the storage — they must outlive the
   * container, in-place writes to them ARE writes to the matrix (plans are
   * never invalidated by value mutation), and their placement is whatever the
   * caller's allocator chose (adoption keeps the split/confinement structure
   * but does NOT re-place bytes; use `from_device` for placed per-shard
   * copies). `contiguous_values()`/`contiguous_colinds()` return the caller's
   * base pointers: the whole arrays are contiguous by construction.
   */
  static sharded_csr adopt(
    place_group& group,
    ::std::int64_t num_rows,
    ::std::int64_t num_cols,
    const int* d_offsets,
    int* d_colinds,
    _Tp* d_values,
    ::std::vector<::std::int64_t> row_boundaries = {})
  {
    ::std::vector<int> h_offsets(static_cast<size_t>(num_rows) + 1);
    cuda_safe_call(cudaMemcpy(h_offsets.data(), d_offsets, h_offsets.size() * sizeof(int), cudaMemcpyDefault));

    sharded_csr m;
    m.rows_ = num_rows;
    m.cols_ = num_cols;
    m.nnz_  = h_offsets[static_cast<size_t>(num_rows)];
    m.init_adopted(group, h_offsets.data(), d_colinds, d_values, mv(row_boundaries));
    return m;
  }

private:
  /// @brief Common construction: offsets are HOST data; colinds/values may be
  /// host or device pointers (the shard copies use `cudaMemcpyDefault`).
  void init(place_group& group,
            const int* h_offsets,
            const int* colinds_src,
            const _Tp* values_src,
            ::std::vector<::std::int64_t> row_boundaries,
            bool contiguous)
  {
    const ::std::int64_t num_rows = rows_;
    const size_t num_shards       = group.size();
    if (num_shards == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "sharded_csr: place group has no places");
    }
    if (row_boundaries.empty())
    {
      row_boundaries = nnz_balanced_boundaries(num_rows, h_offsets, num_shards);
    }
    if (row_boundaries.size() != num_shards - 1)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "sharded_csr: need group.size()-1 row boundaries (" + ::std::to_string(num_shards - 1) + "), got "
                    + ::std::to_string(row_boundaries.size()));
    }

    // Full boundary list [0, b..., num_rows]
    ::std::vector<::std::int64_t> b;
    b.push_back(0);
    b.insert(b.end(), row_boundaries.begin(), row_boundaries.end());
    b.push_back(num_rows);
    for (size_t d = 0; d + 1 < b.size(); d++)
    {
      if (b[d] > b[d + 1] || b[d] < 0 || b[d + 1] > num_rows)
      {
        _CCCL_THROW(::std::invalid_argument, "sharded_csr: row boundaries must be ascending in [0, num_rows]");
      }
    }

    // Allocation specs for the three backing arrays (one shard per place;
    // offsets get rows+1 entries per shard, colinds/values the nnz slice).
    // One stream color for the whole matrix: the shards' reference streams.
    const size_t color = group.next_lane_id();
    ::std::vector<shard_spec> off_specs, nnz_specs;
    for (size_t d = 0; d < num_shards; d++)
    {
      const ::std::int64_t r0 = b[d], r1 = b[d + 1];
      const size_t shard_nnz = static_cast<size_t>(h_offsets[r1] - h_offsets[r0]);
      const auto& eplace     = group.place(d);
      const auto dplace      = eplace.affine_data_place();
      cudaStream_t stream    = group.get_stream(d, color);
      off_specs.emplace_back(static_cast<size_t>(r1 - r0) + 1, dplace, eplace, stream);
      nnz_specs.emplace_back(shard_nnz, dplace, eplace, stream);
    }
    offsets_ = sharded_array<int>::allocate(off_specs);
    // colinds/values shards are exact nnz-slices of the parent arrays, so
    // they admit the contiguous (one-VA-range) backing: unmodified writers
    // then see the whole array through contiguous_values()/_colinds().
    colinds_ = contiguous ? sharded_array<int>::allocate_contiguous(nnz_specs) //
                          : sharded_array<int>::allocate(nnz_specs);
    values_  = contiguous ? sharded_array<_Tp>::allocate_contiguous(nnz_specs) //
                          : sharded_array<_Tp>::allocate(nnz_specs);

    // The slice copies distribute the parent arrays (host or device source:
    // cudaMemcpyDefault). Offsets are rebased per shard first (offsets[0] ==
    // 0 in every shard).
    colinds_.copy_from_host(colinds_src);
    values_.copy_from_host(values_src);
    ::std::vector<int> rebased;
    rebased.reserve(static_cast<size_t>(num_rows) + num_shards);
    for (size_t d = 0; d < num_shards; d++)
    {
      const ::std::int64_t r0 = b[d], r1 = b[d + 1];
      for (::std::int64_t r = r0; r <= r1; r++)
      {
        rebased.push_back(h_offsets[r] - h_offsets[r0]);
      }
    }
    offsets_.copy_from_host(rebased.data());

    // Shard views. Zero-size shards are skipped by sharded_array::allocate,
    // so track the backing shards by walking sizes.
    size_t off_idx = 0, nnz_idx = 0;
    for (size_t d = 0; d < num_shards; d++)
    {
      shard_type sh;
      sh.row_begin = b[d];
      sh.rows      = b[d + 1] - b[d];
      sh.nnz_begin = h_offsets[b[d]];
      sh.nnz       = h_offsets[b[d + 1]] - h_offsets[b[d]];
      sh.place     = group.place(d).affine_data_place();
      sh.exec      = group.place(d);
      sh.stream    = group.get_stream(d, color);
      sh.offsets   = offsets_.shard(off_idx++).data; // rows+1 >= 1, never empty
      if (sh.nnz > 0)
      {
        sh.colinds = colinds_.shard(nnz_idx).data;
        sh.values  = values_.shard(nnz_idx).data;
        nnz_idx++;
      }
      shards_.push_back(sh);
    }
  }

  /// @brief Adoption path: rebased offsets are owned; colinds/values shard
  /// views alias the caller's arrays at their nnz slices.
  void init_adopted(place_group& group,
                    const int* h_offsets,
                    int* colinds_base,
                    _Tp* values_base,
                    ::std::vector<::std::int64_t> row_boundaries)
  {
    const ::std::int64_t num_rows = rows_;
    const size_t num_shards       = group.size();
    if (num_shards == 0)
    {
      _CCCL_THROW(::std::invalid_argument, "sharded_csr: place group has no places");
    }
    if (row_boundaries.empty())
    {
      row_boundaries = nnz_balanced_boundaries(num_rows, h_offsets, num_shards);
    }
    if (row_boundaries.size() != num_shards - 1)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "sharded_csr: need group.size()-1 row boundaries (" + ::std::to_string(num_shards - 1) + "), got "
                    + ::std::to_string(row_boundaries.size()));
    }
    ::std::vector<::std::int64_t> b;
    b.push_back(0);
    b.insert(b.end(), row_boundaries.begin(), row_boundaries.end());
    b.push_back(num_rows);
    for (size_t d = 0; d + 1 < b.size(); d++)
    {
      if (b[d] > b[d + 1] || b[d] < 0 || b[d + 1] > num_rows)
      {
        _CCCL_THROW(::std::invalid_argument, "sharded_csr: row boundaries must be ascending in [0, num_rows]");
      }
    }

    // Owned rebased offsets only (per-place placement as usual).
    const size_t color = group.next_lane_id();
    ::std::vector<shard_spec> off_specs;
    for (size_t d = 0; d < num_shards; d++)
    {
      const auto& eplace = group.place(d);
      off_specs.emplace_back(
        static_cast<size_t>(b[d + 1] - b[d]) + 1, eplace.affine_data_place(), eplace, group.get_stream(d, color));
    }
    offsets_ = sharded_array<int>::allocate(off_specs);
    ::std::vector<int> rebased;
    rebased.reserve(static_cast<size_t>(num_rows) + num_shards);
    for (size_t d = 0; d < num_shards; d++)
    {
      for (::std::int64_t r = b[d]; r <= b[d + 1]; r++)
      {
        rebased.push_back(h_offsets[r] - h_offsets[b[d]]);
      }
    }
    offsets_.copy_from_host(rebased.data());

    // Shard views alias the caller's arrays at their nnz slices.
    for (size_t d = 0; d < num_shards; d++)
    {
      shard_type sh;
      sh.row_begin = b[d];
      sh.rows      = b[d + 1] - b[d];
      sh.nnz_begin = h_offsets[b[d]];
      sh.nnz       = h_offsets[b[d + 1]] - h_offsets[b[d]];
      sh.place     = group.place(d).affine_data_place();
      sh.exec      = group.place(d);
      sh.stream    = group.get_stream(d, color);
      sh.offsets   = offsets_.shard(d).data;
      sh.colinds   = colinds_base + sh.nnz_begin;
      sh.values    = values_base + sh.nnz_begin;
      shards_.push_back(sh);
    }
    adopted_colinds_ = colinds_base;
    adopted_values_  = values_base;
  }

  sharded_csr() = default; // used by the adopt factory

public:
  /// @brief True when the whole values array is reachable through one base
  /// pointer: contiguous (VMM) backing, or adopted caller arrays.
  bool values_contiguous() const
  {
    return adopted_values_ != nullptr || values_.is_contiguous();
  }

  /// @brief Base pointer of the whole values array (`nullptr` unless built
  /// with `contiguous = true`). In-place mutation through it (or through the
  /// shard views) never invalidates the per-shard library plans.
  _Tp* contiguous_values() const
  {
    return adopted_values_ != nullptr ? adopted_values_ : values_.contiguous_data();
  }

  /// @brief Base pointer of the whole colinds array (`nullptr` unless built
  /// with `contiguous = true`).
  int* contiguous_colinds() const
  {
    return adopted_colinds_ != nullptr ? adopted_colinds_ : colinds_.contiguous_data();
  }

  // The container owns library state whose destruction order matters; keep it
  // move-only like the other owning sharded containers.
  sharded_csr(sharded_csr&&)                 = default;
  sharded_csr& operator=(sharded_csr&&)      = default;
  sharded_csr(const sharded_csr&)            = delete;
  sharded_csr& operator=(const sharded_csr&) = delete;

  /**
   * @brief Default split: shard d starts at the first row whose cumulative
   *        nnz reaches d*nnz/num_shards. Every shard keeps at least one row.
   */
  static ::std::vector<::std::int64_t>
  nnz_balanced_boundaries(::std::int64_t num_rows, const int* h_offsets, size_t num_shards)
  {
    ::std::vector<::std::int64_t> bounds;
    const ::std::int64_t nnz = h_offsets[num_rows];
    ::std::int64_t prev      = 0;
    for (size_t d = 1; d < num_shards; d++)
    {
      const ::std::int64_t target = static_cast<::std::int64_t>(d) * nnz / static_cast<::std::int64_t>(num_shards);
      const int* lo               = ::std::lower_bound(h_offsets, h_offsets + num_rows + 1, static_cast<int>(target));
      ::std::int64_t row          = lo - h_offsets;
      // Keep every shard non-empty in rows
      row  = ::std::min(::std::max(row, prev + 1), num_rows - static_cast<::std::int64_t>(num_shards - d));
      prev = row;
      bounds.push_back(row);
    }
    return bounds;
  }

  /**
   * @brief Time-balanced split from measured per-shard times.
   *
   * Under SM confinement a split finishes at max(shard time), so the split
   * must balance *time*, not nnz. Given the per-shard times measured with the
   * CURRENT boundaries (e.g. `spmv_shard_times` / `spmm_shard_times` from
   * `<cuda/experimental/sharded_sparse.cuh>`), this models each current shard
   * as a constant nnz-throughput region (rate_d = nnz_d / ms_d), predicts the
   * time of any row range as the rate-weighted nnz it overlaps, and places
   * new boundaries so every new shard's predicted time is total/num_shards.
   *
   * One round removes most of the imbalance; rates shift as rows change
   * shards, so iterate with fresh measurements (keeping the best measured
   * split) when the matrix is reused enough to amortize it.
   *
   * @param num_rows Number of rows of the operator
   * @param h_offsets Host CSR offsets (num_rows+1)
   * @param current_boundaries Interior boundaries the times were measured
   *        with (size num_shards-1; the container's shard(d+1).row_begin)
   * @param shard_ms Measured per-shard times (size num_shards)
   * @return New interior boundaries (size num_shards-1)
   */
  static ::std::vector<::std::int64_t> time_balanced_boundaries(
    ::std::int64_t num_rows,
    const int* h_offsets,
    const ::std::vector<::std::int64_t>& current_boundaries,
    const ::std::vector<double>& shard_ms)
  {
    const size_t num_shards = shard_ms.size();
    if (current_boundaries.size() != num_shards - 1)
    {
      _CCCL_THROW(::std::invalid_argument, "time_balanced_boundaries: boundaries/times size mismatch");
    }
    // Full boundary list of the measured split + per-old-shard cost per nnz
    ::std::vector<::std::int64_t> b;
    b.push_back(0);
    b.insert(b.end(), current_boundaries.begin(), current_boundaries.end());
    b.push_back(num_rows);
    ::std::vector<double> cost_per_nnz(num_shards);
    double total_ms = 0;
    for (size_t d = 0; d < num_shards; d++)
    {
      const ::std::int64_t nnz_d = h_offsets[b[d + 1]] - h_offsets[b[d]];
      cost_per_nnz[d]            = (nnz_d > 0) ? shard_ms[d] / static_cast<double>(nnz_d) : 0.0;
      total_ms += shard_ms[d];
    }
    const double target = total_ms / static_cast<double>(num_shards);

    ::std::vector<::std::int64_t> bounds;
    double acc          = 0.0; // predicted ms accumulated in the current new shard
    size_t old_d        = 0;
    ::std::int64_t prev = 0;
    for (::std::int64_t r = 0; r < num_rows && bounds.size() < num_shards - 1; r++)
    {
      while (old_d + 1 < num_shards && r >= b[old_d + 1])
      {
        old_d++;
      }
      acc += cost_per_nnz[old_d] * static_cast<double>(h_offsets[r + 1] - h_offsets[r]);
      if (acc >= target)
      {
        ::std::int64_t row = r + 1;
        row =
          ::std::min(::std::max(row, prev + 1), num_rows - static_cast<::std::int64_t>(num_shards - 1 - bounds.size()));
        bounds.push_back(row);
        prev = row;
        acc  = 0.0;
      }
    }
    while (bounds.size() < num_shards - 1)
    {
      // Degenerate tail: keep remaining shards non-empty
      ::std::int64_t row = ::std::min(prev + 1, num_rows - static_cast<::std::int64_t>(num_shards - 1 - bounds.size()));
      bounds.push_back(row);
      prev = row;
    }
    return bounds;
  }

  /// @brief This matrix's interior row boundaries (shard(d).row_begin for
  /// d in [1, num_shards)), i.e. the `current_boundaries` argument of
  /// `time_balanced_boundaries`.
  ::std::vector<::std::int64_t> interior_boundaries() const
  {
    ::std::vector<::std::int64_t> bounds;
    for (size_t d = 1; d < shards_.size(); d++)
    {
      bounds.push_back(shards_[d].row_begin);
    }
    return bounds;
  }

  /**
   * @brief Allocate a row-partitioned dense output matching this matrix
   *        (n_cols columns, row-major): shard d holds rows [row_begin,
   *        row_begin + rows) in that shard's place. Row partition => the
   *        output shards are disjoint, so no combine step is ever needed.
   *
   * @param n_cols Dense columns (row-major, ld = n_cols)
   * @param contiguous When true, the shards are views into ONE contiguous VA
   *        range (`sharded_array::allocate_contiguous`): unmodified
   *        downstream consumers read the whole result through
   *        `contiguous_data()` while each row block keeps (granule-
   *        approximate) per-place physical placement. Logical row boundaries
   *        stay exact either way.
   */
  sharded_array<_Tp> make_row_partitioned(::std::int64_t n_cols = 1, bool contiguous = false) const
  {
    ::std::vector<shard_spec> specs;
    for (const auto& sh : shards_)
    {
      specs.emplace_back(static_cast<size_t>(sh.rows * n_cols), sh.place, sh.exec, sh.stream);
    }
    return contiguous ? sharded_array<_Tp>::allocate_contiguous(specs) : sharded_array<_Tp>::allocate(specs);
  }

  ::std::int64_t num_rows() const
  {
    return rows_;
  }
  ::std::int64_t num_cols() const
  {
    return cols_;
  }
  ::std::int64_t nnz() const
  {
    return nnz_;
  }
  size_t num_shards() const
  {
    return shards_.size();
  }
  shard_type& shard(size_t idx)
  {
    _CCCL_ASSERT(idx < shards_.size(), "sharded_csr: shard index out of range");
    return shards_[idx];
  }
  const shard_type& shard(size_t idx) const
  {
    _CCCL_ASSERT(idx < shards_.size(), "sharded_csr: shard index out of range");
    return shards_[idx];
  }

  // ========== Stream ordering: composing with a caller stream ==========

  /**
   * @brief Declare that subsequent work on every shard stream depends on the
   *        work currently enqueued on @p stream (fork a caller stream out to
   *        the matrix's per-shard streams).
   *
   * ORDERING DECLARATION, NOT A SYNCHRONIZATION: one event is recorded on
   * @p stream and every shard stream waits on it; the host returns
   * immediately. Because the three backing arrays (offsets/colinds/values)
   * share the matrix's per-place reference streams, ordering the shard
   * streams orders every consumer of the matrix. Same pooled-event mechanics
   * and capture behavior as `sharded_array<T>::fork_from` (see there and
   * `reserved::fork_join_event_pool`).
   */
  void fork_from(cudaStream_t stream) const
  {
    cudaEvent_t event = nullptr; // recorded once, on the first distinct shard stream
    for (const auto& sh : shards_)
    {
      if (!sh.stream || sh.stream == stream)
      {
        continue;
      }
      if (!event)
      {
        // Capture-safe on CTK >= 12.8 (probed: cudaStreamGetDevice answers
        // during capture); pre-12.8 falls back to the current device, which
        // is correct on single-device systems (see get_device_from_stream).
        const int device = places::get_device_from_stream(stream);
        event            = fork_join_events_.fork_event(device);
        cuda_safe_call(cudaEventRecord(event, stream));
      }
      cuda_safe_call(cudaStreamWaitEvent(sh.stream, event, 0));
    }
  }

  /**
   * @brief Declare that subsequent work on @p stream depends on the work
   *        currently enqueued on every shard stream (join the matrix's
   *        per-shard streams back into a caller stream).
   *
   * ORDERING DECLARATION, NOT A SYNCHRONIZATION: an event is recorded on each
   * shard stream and @p stream waits on all of them; the host returns
   * immediately. The mirror image of `fork_from`, with the same pooled-event
   * mechanics and capture behavior as `sharded_array<T>::join_into`.
   */
  void join_into(cudaStream_t stream) const
  {
    for (size_t i = 0; i < shards_.size(); i++)
    {
      const auto& sh = shards_[i];
      if (!sh.stream || sh.stream == stream)
      {
        continue;
      }
      cudaEvent_t event = nullptr;
      {
        // Create/record in the shard's context so the event matches its stream.
        exec_place_scope scope(sh.exec);
        event = fork_join_events_.join_event(i);
        cuda_safe_call(cudaEventRecord(event, sh.stream));
      }
      cuda_safe_call(cudaStreamWaitEvent(stream, event, 0));
    }
  }

private:
  ::std::int64_t rows_ = 0, cols_ = 0, nnz_ = 0;
  ::std::vector<shard_type> shards_;
  // Backing storage (owning); shards_ hold raw views into these
  sharded_array<int> offsets_;
  sharded_array<int> colinds_; // empty in the adopted grade
  sharded_array<_Tp> values_; // empty in the adopted grade
  int* adopted_colinds_ = nullptr; // caller-owned bases when adopted
  _Tp* adopted_values_  = nullptr;
  // Pooled events for fork_from/join_into (lazily created; mutable because
  // the ordering declarations are const -- they do not modify the matrix).
  mutable reserved::fork_join_event_pool fork_join_events_;
};
} // namespace cuda::experimental::sharded
