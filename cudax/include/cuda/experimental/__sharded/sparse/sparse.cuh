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
 * @brief Placement-localized sparse products over `sharded_csr` with cuSPARSE
 *        as the per-place engine: `sharded::spmv` / `sharded::spmm` over
 *        EXPLICIT caller-held state (`cusparse_handles`, `spmv_plan` /
 *        `spmm_plan`), plus the measured-rebalance utilities
 *        `spmv_shard_times` / `spmm_shard_times`.
 *
 * OPT-IN VENDOR HEADER — not included by `<cuda/experimental/sharded.cuh>`
 * (same model as the cuRAND tier in `random.cuh`); consumers link cuSPARSE.
 * The `sharded_csr` container never depends on this header.
 *
 * The structure is the ladder's usual two tiers with a closed library in the
 * engine slot: the container owns the row partition and placement; the engine
 * tier is ONE confined cuSPARSE call per shard, on the shard's place stream.
 * A row partition makes the output row blocks disjoint, so there is never a
 * combine step.
 *
 * LIBRARY STATE IS EXPLICIT, split by its natural scope, and the caller can
 * read every lifetime off the page:
 *  - `cusparse_handles` — PLACE-BOUND: one `cusparseHandle_t` per place of a
 *    group, created lazily under the place's exec scope, shared by every
 *    matrix and plan built over it. Create it once next to the group; it must
 *    outlive the plans. (Creating handles per call measurably serializes on
 *    the host — the cuRAND tier's generator-lifecycle lesson.)
 *  - `spmv_plan` / `spmm_plan` — MATRIX-BOUND: per-shard descriptors,
 *    workspace and preprocessed plan, built lazily on first run against the
 *    shard's fixed addresses, reused across calls (later calls only rebind
 *    the dense pointers and the handle's stream). The plan references its
 *    matrix and handles: both must outlive it.
 *
 * Dense operands are plain device pointers readable from every place (for
 * example one whole-device allocation): which COPIES of a re-read operand
 * should exist, and when they go stale, is a coherence question that belongs
 * to the binding tier. Outputs are row-partitioned `sharded_array`s
 * (contiguous backing included), typically `sharded_csr::make_row_partitioned`.
 *
 * ASYNCHRONOUS with respect to the host: work is enqueued on the shards'
 * streams; join with `barrier(...)`, the output's `join_into`, or the group.
 * Not thread-safe per plan (mutable per-matrix state): externally serialize
 * concurrent calls on the same plan.
 *
 * Umbrella over `sparse/cusparse.cuh` (shared handles and helpers),
 * `sparse/spmv.cuh`, `sparse/spmm.cuh` and `sparse/rebalance.cuh`.
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

#include <cuda/experimental/__sharded/sparse/cusparse.cuh>
#include <cuda/experimental/__sharded/sparse/rebalance.cuh>
#include <cuda/experimental/__sharded/sparse/spmm.cuh>
#include <cuda/experimental/__sharded/sparse/spmv.cuh>
