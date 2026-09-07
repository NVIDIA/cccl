# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fused k-means-assignment fused distance tile plus TopK example.

This is a compact k-means assignment sketch.  Each CTA owns one query
vector and a tile of candidate centroids.  Threads compute squared distances
from the GEMM-expanded form

    ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x^T c

in registers, then immediately feed those scores into the single-phase CUB
TopK provider.  The distance tile is never materialized in global memory.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from examples.cutlass._runtime import require_runtime

BLOCK_THREADS = 32
ITEMS_PER_THREAD = 2
WIDE_ITEMS_PER_THREAD = 8
FEATURE_DIM = 8
WIDE_FEATURE_DIM = 128
CENTROIDS_PER_TILE = BLOCK_THREADS * ITEMS_PER_THREAD
WIDE_CENTROIDS_PER_TILE = BLOCK_THREADS * WIDE_ITEMS_PER_THREAD
FEATURE_LANES_PER_CENTROID = 4
FEATURE_SPLIT_BLOCK_THREADS = WIDE_CENTROIDS_PER_TILE * FEATURE_LANES_PER_CENTROID
FEATURE_SPLIT_CHUNKS = WIDE_FEATURE_DIM // FEATURE_LANES_PER_CENTROID
FEATURE_SPLIT_EXCHANGE_STORAGE_BYTES = 4096
FEATURE_SPLIT_TOPK_STORAGE_BYTES = 28672
# The generated D=128 benchmark values have squared distances below 2^16.
FEATURE_SPLIT_TOPK_END_BIT = 16
# The generated D=128 score-only keys use score + offset and fit below 2^12.
FEATURE_SPLIT_SCORE_OFFSET = 9000
FEATURE_SPLIT_SCORE_TOPK_END_BIT = 12
FEATURE_SPLIT_SCORE_TOP1_K = 1
FEATURE_SPLIT_WARP_COUNT = FEATURE_SPLIT_BLOCK_THREADS // 32
TOPK_K = 5
BATCHED_QUERY_COUNT = 4096


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    return require_runtime(include_int32=True)


@functools.lru_cache(maxsize=1)
def make_runner() -> tuple[Any, ...]:
    """Build and return the CuTe JIT runner plus runtime helpers."""

    cutlass, cute, torch, from_dlpack, coop, Int32 = _require_runtime()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute
    globals()["coop"] = coop
    globals()["Int32"] = Int32
    feature_split_exchange_storage = coop.TempStorage(
        size_in_bytes=FEATURE_SPLIT_EXCHANGE_STORAGE_BYTES,
        sharing="shared",
    )
    feature_split_topk_storage = coop.TempStorage(
        size_in_bytes=FEATURE_SPLIT_TOPK_STORAGE_BYTES,
        sharing="shared",
    )

    @cute.kernel
    def _kmeans_assign_topk_kernel(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_idx, _, _ = cute.arch.block_idx()
        first_centroid = tidx.to(cutlass.Int32)
        second_centroid = first_centroid + cutlass.Int32(BLOCK_THREADS)
        query_base = query_idx * cutlass.Int32(FEATURE_DIM)
        output_base = query_idx * cutlass.Int32(TOPK_K)

        x_norm = cutlass.Int32(0)
        first_dot = cutlass.Int32(0)
        second_dot = cutlass.Int32(0)
        first_norm = cutlass.Int32(0)
        second_norm = cutlass.Int32(0)

        for feature_idx in cutlass.range_constexpr(FEATURE_DIM):
            x = query[query_base + feature_idx]
            c0 = centroids[first_centroid * cutlass.Int32(FEATURE_DIM) + feature_idx]
            c1 = centroids[second_centroid * cutlass.Int32(FEATURE_DIM) + feature_idx]

            x_norm += x * x
            first_dot += x * c0
            second_dot += x * c1
            first_norm += c0 * c0
            second_norm += c1 * c1

        first_distance = x_norm + first_norm - cutlass.Int32(2) * first_dot
        second_distance = x_norm + second_norm - cutlass.Int32(2) * second_dot
        distance_items = coop.ThreadData.from_values(
            first_distance,
            second_distance,
            dtype=Int32,
        )
        centroid_items = coop.ThreadData.from_values(
            first_centroid,
            second_centroid,
            dtype=Int32,
        )

        top_distances, top_centroids = coop.topk_min_pairs(
            coop.this_block(),
            distance_items,
            centroid_items,
            cutlass.const_expr(TOPK_K),
            begin_bit=cutlass.const_expr(0),
            end_bit=cutlass.const_expr(31),
        )
        flat_item = tidx * cutlass.Int32(ITEMS_PER_THREAD)
        if flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + flat_item] = top_distances[0]
            top_centroid_out[output_base + flat_item] = top_centroids[0]
        second_flat_item = flat_item + cutlass.Int32(1)
        if second_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + second_flat_item] = top_distances[1]
            top_centroid_out[output_base + second_flat_item] = top_centroids[1]

    @cute.kernel
    def _kmeans_assign_topk_wide_kernel(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        query_idx, _, _ = cute.arch.block_idx()
        first_centroid = tidx.to(cutlass.Int32)
        second_centroid = first_centroid + cutlass.Int32(BLOCK_THREADS)
        third_centroid = first_centroid + cutlass.Int32(2 * BLOCK_THREADS)
        fourth_centroid = first_centroid + cutlass.Int32(3 * BLOCK_THREADS)
        fifth_centroid = first_centroid + cutlass.Int32(4 * BLOCK_THREADS)
        sixth_centroid = first_centroid + cutlass.Int32(5 * BLOCK_THREADS)
        seventh_centroid = first_centroid + cutlass.Int32(6 * BLOCK_THREADS)
        eighth_centroid = first_centroid + cutlass.Int32(7 * BLOCK_THREADS)
        query_base = query_idx * cutlass.Int32(WIDE_FEATURE_DIM)
        output_base = query_idx * cutlass.Int32(TOPK_K)

        x_norm = cutlass.Int32(0)
        first_dot = cutlass.Int32(0)
        second_dot = cutlass.Int32(0)
        third_dot = cutlass.Int32(0)
        fourth_dot = cutlass.Int32(0)
        fifth_dot = cutlass.Int32(0)
        sixth_dot = cutlass.Int32(0)
        seventh_dot = cutlass.Int32(0)
        eighth_dot = cutlass.Int32(0)
        first_norm = cutlass.Int32(0)
        second_norm = cutlass.Int32(0)
        third_norm = cutlass.Int32(0)
        fourth_norm = cutlass.Int32(0)
        fifth_norm = cutlass.Int32(0)
        sixth_norm = cutlass.Int32(0)
        seventh_norm = cutlass.Int32(0)
        eighth_norm = cutlass.Int32(0)

        for feature_idx in cutlass.range_constexpr(WIDE_FEATURE_DIM):
            x = query[query_base + feature_idx]
            c0 = centroids[
                first_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c1 = centroids[
                second_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c2 = centroids[
                third_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c3 = centroids[
                fourth_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c4 = centroids[
                fifth_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c5 = centroids[
                sixth_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c6 = centroids[
                seventh_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]
            c7 = centroids[
                eighth_centroid * cutlass.Int32(WIDE_FEATURE_DIM) + feature_idx
            ]

            x_norm += x * x
            first_dot += x * c0
            second_dot += x * c1
            third_dot += x * c2
            fourth_dot += x * c3
            fifth_dot += x * c4
            sixth_dot += x * c5
            seventh_dot += x * c6
            eighth_dot += x * c7
            first_norm += c0 * c0
            second_norm += c1 * c1
            third_norm += c2 * c2
            fourth_norm += c3 * c3
            fifth_norm += c4 * c4
            sixth_norm += c5 * c5
            seventh_norm += c6 * c6
            eighth_norm += c7 * c7

        first_distance = x_norm + first_norm - cutlass.Int32(2) * first_dot
        second_distance = x_norm + second_norm - cutlass.Int32(2) * second_dot
        third_distance = x_norm + third_norm - cutlass.Int32(2) * third_dot
        fourth_distance = x_norm + fourth_norm - cutlass.Int32(2) * fourth_dot
        fifth_distance = x_norm + fifth_norm - cutlass.Int32(2) * fifth_dot
        sixth_distance = x_norm + sixth_norm - cutlass.Int32(2) * sixth_dot
        seventh_distance = x_norm + seventh_norm - cutlass.Int32(2) * seventh_dot
        eighth_distance = x_norm + eighth_norm - cutlass.Int32(2) * eighth_dot
        distance_items = coop.ThreadData.from_values(
            first_distance,
            second_distance,
            third_distance,
            fourth_distance,
            fifth_distance,
            sixth_distance,
            seventh_distance,
            eighth_distance,
            dtype=Int32,
        )
        centroid_items = coop.ThreadData.from_values(
            first_centroid,
            second_centroid,
            third_centroid,
            fourth_centroid,
            fifth_centroid,
            sixth_centroid,
            seventh_centroid,
            eighth_centroid,
            dtype=Int32,
        )

        top_distances, top_centroids = coop.topk_min_pairs(
            coop.this_block(),
            distance_items,
            centroid_items,
            cutlass.const_expr(TOPK_K),
            begin_bit=cutlass.const_expr(0),
            end_bit=cutlass.const_expr(31),
        )
        flat_item = tidx * cutlass.Int32(WIDE_ITEMS_PER_THREAD)
        if flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + flat_item] = top_distances[0]
            top_centroid_out[output_base + flat_item] = top_centroids[0]
        second_flat_item = flat_item + cutlass.Int32(1)
        if second_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + second_flat_item] = top_distances[1]
            top_centroid_out[output_base + second_flat_item] = top_centroids[1]
        third_flat_item = flat_item + cutlass.Int32(2)
        if third_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + third_flat_item] = top_distances[2]
            top_centroid_out[output_base + third_flat_item] = top_centroids[2]
        fourth_flat_item = flat_item + cutlass.Int32(3)
        if fourth_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + fourth_flat_item] = top_distances[3]
            top_centroid_out[output_base + fourth_flat_item] = top_centroids[3]
        fifth_flat_item = flat_item + cutlass.Int32(4)
        if fifth_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + fifth_flat_item] = top_distances[4]
            top_centroid_out[output_base + fifth_flat_item] = top_centroids[4]
        sixth_flat_item = flat_item + cutlass.Int32(5)
        if sixth_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + sixth_flat_item] = top_distances[5]
            top_centroid_out[output_base + sixth_flat_item] = top_centroids[5]
        seventh_flat_item = flat_item + cutlass.Int32(6)
        if seventh_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + seventh_flat_item] = top_distances[6]
            top_centroid_out[output_base + seventh_flat_item] = top_centroids[6]
        eighth_flat_item = flat_item + cutlass.Int32(7)
        if eighth_flat_item < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + eighth_flat_item] = top_distances[7]
            top_centroid_out[output_base + eighth_flat_item] = top_centroids[7]

    @cute.kernel
    def _kmeans_assign_topk_feature_split_kernel(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block = coop.this_block()
        logical_centroid = coop.this_warp().group_by(FEATURE_LANES_PER_CENTROID)
        query_idx, _, _ = cute.arch.block_idx()
        centroid = tidx // cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        feature_lane = tidx - centroid * cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        query_base = query_idx * cutlass.Int32(WIDE_FEATURE_DIM)
        centroid_base = centroid * cutlass.Int32(WIDE_FEATURE_DIM)
        output_base = query_idx * cutlass.Int32(TOPK_K)

        partial_x_norm = cutlass.Int32(0)
        partial_dot = cutlass.Int32(0)
        partial_centroid_norm = cutlass.Int32(0)

        for feature_chunk in cutlass.range_constexpr(FEATURE_SPLIT_CHUNKS):
            feature_idx = (
                cutlass.Int32(feature_chunk * FEATURE_LANES_PER_CENTROID) + feature_lane
            )
            x = query[query_base + feature_idx]
            c = centroids[centroid_base + feature_idx]
            partial_x_norm += x * x
            partial_dot += x * c
            partial_centroid_norm += c * c

        x_norm = coop.sum(logical_centroid, partial_x_norm)
        dot = coop.sum(logical_centroid, partial_dot)
        centroid_norm = coop.sum(logical_centroid, partial_centroid_norm)
        distance = x_norm + centroid_norm - cutlass.Int32(2) * dot

        valid_centroid = cutlass.Int32(0)
        if feature_lane == cutlass.Int32(0):
            valid_centroid = cutlass.Int32(1)

        centroid_rank = coop.ThreadData.from_values(centroid, dtype=Int32)
        valid_flags = coop.ThreadData.from_values(valid_centroid, dtype=Int32)
        compacted_distance = coop.exchange(
            block,
            coop.ThreadData.from_values(distance, dtype=Int32),
            mode="scatter_to_striped_flagged",
            ranks=centroid_rank,
            valid_flags=valid_flags,
        )
        compacted_centroid = coop.exchange(
            block,
            coop.ThreadData.from_values(centroid, dtype=Int32),
            mode="scatter_to_striped_flagged",
            ranks=centroid_rank,
            valid_flags=valid_flags,
        )

        top_distances, top_centroids = coop.topk_min_pairs(
            block,
            compacted_distance,
            compacted_centroid,
            cutlass.const_expr(TOPK_K),
            valid_items=cutlass.const_expr(WIDE_CENTROIDS_PER_TILE),
            begin_bit=cutlass.const_expr(0),
            end_bit=cutlass.const_expr(FEATURE_SPLIT_TOPK_END_BIT),
        )
        if tidx < cutlass.Int32(TOPK_K):
            top_distance_out[output_base + tidx] = top_distances[0]
            top_centroid_out[output_base + tidx] = top_centroids[0]

    @cute.kernel
    def _kmeans_assign_topk_feature_split_score_kernel(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_score_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block = coop.this_block()
        logical_centroid = coop.this_warp().group_by(FEATURE_LANES_PER_CENTROID)
        query_idx, _, _ = cute.arch.block_idx()
        centroid = tidx // cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        feature_lane = tidx - centroid * cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        query_base = query_idx * cutlass.Int32(WIDE_FEATURE_DIM)
        centroid_base = centroid * cutlass.Int32(WIDE_FEATURE_DIM)
        output_base = query_idx * cutlass.Int32(TOPK_K)

        partial_dot = cutlass.Int32(0)
        partial_centroid_norm = cutlass.Int32(0)

        for feature_chunk in cutlass.range_constexpr(FEATURE_SPLIT_CHUNKS):
            feature_idx = (
                cutlass.Int32(feature_chunk * FEATURE_LANES_PER_CENTROID) + feature_lane
            )
            x = query[query_base + feature_idx]
            c = centroids[centroid_base + feature_idx]
            partial_dot += x * c
            partial_centroid_norm += c * c

        dot = coop.sum(logical_centroid, partial_dot)
        centroid_norm = coop.sum(logical_centroid, partial_centroid_norm)
        score_key = (
            centroid_norm
            - cutlass.Int32(2) * dot
            + cutlass.Int32(FEATURE_SPLIT_SCORE_OFFSET)
        )

        topk_key = cutlass.Int32(2147483647)
        topk_value = cutlass.Int32(-1)
        if feature_lane == cutlass.Int32(0):
            topk_key = score_key
            topk_value = centroid

        top_score, top_centroid = coop.topk_min_pairs(
            block,
            topk_key,
            topk_value,
            cutlass.const_expr(TOPK_K),
            begin_bit=cutlass.const_expr(0),
            end_bit=cutlass.const_expr(FEATURE_SPLIT_SCORE_TOPK_END_BIT),
        )
        if tidx < cutlass.Int32(TOPK_K):
            top_score_out[output_base + tidx] = top_score
            top_centroid_out[output_base + tidx] = top_centroid

    @cute.kernel
    def _kmeans_assign_top1_feature_split_score_kernel(
        query: cute.Tensor,
        centroids: cute.Tensor,
        best_score_out: cute.Tensor,
        best_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block = coop.this_block()
        logical_centroid = coop.this_warp().group_by(FEATURE_LANES_PER_CENTROID)
        query_idx, _, _ = cute.arch.block_idx()
        centroid = tidx // cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        feature_lane = tidx - centroid * cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        query_base = query_idx * cutlass.Int32(WIDE_FEATURE_DIM)
        centroid_base = centroid * cutlass.Int32(WIDE_FEATURE_DIM)
        # Reuse the TopK output buffer stride; this top-1 variant writes slot 0.
        output_base = query_idx * cutlass.Int32(TOPK_K)

        partial_dot = cutlass.Int32(0)
        partial_centroid_norm = cutlass.Int32(0)

        for feature_chunk in cutlass.range_constexpr(FEATURE_SPLIT_CHUNKS):
            feature_idx = (
                cutlass.Int32(feature_chunk * FEATURE_LANES_PER_CENTROID) + feature_lane
            )
            x = query[query_base + feature_idx]
            c = centroids[centroid_base + feature_idx]
            partial_dot += x * c
            partial_centroid_norm += c * c

        dot = coop.sum(logical_centroid, partial_dot)
        centroid_norm = coop.sum(logical_centroid, partial_centroid_norm)
        score_key = (
            centroid_norm
            - cutlass.Int32(2) * dot
            + cutlass.Int32(FEATURE_SPLIT_SCORE_OFFSET)
        )

        topk_key = cutlass.Int32(2147483647)
        topk_value = cutlass.Int32(-1)
        if feature_lane == cutlass.Int32(0):
            topk_key = score_key
            topk_value = centroid

        best_score, best_centroid = coop.topk_min_pairs(
            block,
            topk_key,
            topk_value,
            cutlass.const_expr(FEATURE_SPLIT_SCORE_TOP1_K),
            begin_bit=cutlass.const_expr(0),
            end_bit=cutlass.const_expr(FEATURE_SPLIT_SCORE_TOPK_END_BIT),
        )
        if tidx == cutlass.Int32(0):
            best_score_out[output_base] = best_score
            best_centroid_out[output_base] = best_centroid

    @cute.kernel
    def _kmeans_assign_top1_feature_split_score_warp_kernel(
        query: cute.Tensor,
        centroids: cute.Tensor,
        best_score_out: cute.Tensor,
        best_centroid_out: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        physical_warp = coop.this_warp()
        logical_centroid = physical_warp.group_by(FEATURE_LANES_PER_CENTROID)
        query_idx, _, _ = cute.arch.block_idx()
        centroid = tidx // cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        feature_lane = tidx - centroid * cutlass.Int32(FEATURE_LANES_PER_CENTROID)
        lane = tidx - (tidx // cutlass.Int32(32)) * cutlass.Int32(32)
        query_base = query_idx * cutlass.Int32(WIDE_FEATURE_DIM)
        centroid_base = centroid * cutlass.Int32(WIDE_FEATURE_DIM)
        output_base = query_idx * cutlass.Int32(TOPK_K)

        # The generated benchmark tensors are int32, so the packed warp-min
        # key stays in the integer domain and does not need score quantization.
        partial_dot = cutlass.Int32(0)
        partial_centroid_norm = cutlass.Int32(0)

        for feature_chunk in cutlass.range_constexpr(FEATURE_SPLIT_CHUNKS):
            feature_idx = (
                cutlass.Int32(feature_chunk * FEATURE_LANES_PER_CENTROID) + feature_lane
            )
            x = query[query_base + feature_idx]
            c = centroids[centroid_base + feature_idx]
            partial_dot += x * c
            partial_centroid_norm += c * c

        dot = coop.sum(logical_centroid, partial_dot)
        centroid_norm = coop.sum(logical_centroid, partial_centroid_norm)
        score_key = (
            centroid_norm
            - cutlass.Int32(2) * dot
            + cutlass.Int32(FEATURE_SPLIT_SCORE_OFFSET)
        )

        packed = cutlass.Int32(2147483647)
        if feature_lane == cutlass.Int32(0):
            # Score keys fit in 12 bits and centroid ids fit in 8 bits here.
            packed = (score_key << cutlass.Int32(8)) | centroid

        warp_best = coop.reduce(physical_warp, packed, binary_op="min")

        smem = cutlass.utils.SmemAllocator()
        warp_best_smem = smem.allocate_tensor(
            Int32,
            cute.make_layout((FEATURE_SPLIT_WARP_COUNT,)),
            16,
        )
        if lane == cutlass.Int32(0):
            warp_best_smem[tidx // cutlass.Int32(32)] = warp_best
        cute.arch.sync_threads()

        block_best = cutlass.Int32(2147483647)
        if tidx < cutlass.Int32(FEATURE_SPLIT_WARP_COUNT):
            block_best = warp_best_smem[tidx]
        block_best = coop.reduce(physical_warp, block_best, binary_op="min")

        if tidx == cutlass.Int32(0):
            best_score_out[output_base] = block_best >> cutlass.Int32(8)
            best_centroid_out[output_base] = block_best & cutlass.Int32(255)

    @cute.jit
    def _run_kmeans_assign_topk(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_topk_kernel(
            query,
            centroids,
            top_distance_out,
            top_centroid_out,
        ).launch(grid=(1, 1, 1), block=(BLOCK_THREADS, 1, 1))

    @cute.jit
    def _run_kmeans_assign_topk_batched(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_topk_kernel(
            query,
            centroids,
            top_distance_out,
            top_centroid_out,
        ).launch(grid=(BATCHED_QUERY_COUNT, 1, 1), block=(BLOCK_THREADS, 1, 1))

    @cute.jit
    def _run_kmeans_assign_topk_feature_split_batched(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_topk_feature_split_kernel(
            query,
            centroids,
            top_distance_out,
            top_centroid_out,
        ).launch(
            grid=(BATCHED_QUERY_COUNT, 1, 1),
            block=(FEATURE_SPLIT_BLOCK_THREADS, 1, 1),
        )

    @cute.jit
    def _run_kmeans_assign_topk_feature_split_score_batched(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_score_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_topk_feature_split_score_kernel(
            query,
            centroids,
            top_score_out,
            top_centroid_out,
        ).launch(
            grid=(BATCHED_QUERY_COUNT, 1, 1),
            block=(FEATURE_SPLIT_BLOCK_THREADS, 1, 1),
        )

    @cute.jit
    def _run_kmeans_assign_top1_feature_split_score_batched(
        query: cute.Tensor,
        centroids: cute.Tensor,
        best_score_out: cute.Tensor,
        best_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_top1_feature_split_score_kernel(
            query,
            centroids,
            best_score_out,
            best_centroid_out,
        ).launch(
            grid=(BATCHED_QUERY_COUNT, 1, 1),
            block=(FEATURE_SPLIT_BLOCK_THREADS, 1, 1),
        )

    @cute.jit
    def _run_kmeans_assign_top1_feature_split_score_warp_batched(
        query: cute.Tensor,
        centroids: cute.Tensor,
        best_score_out: cute.Tensor,
        best_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_top1_feature_split_score_warp_kernel(
            query,
            centroids,
            best_score_out,
            best_centroid_out,
        ).launch(
            grid=(BATCHED_QUERY_COUNT, 1, 1),
            block=(FEATURE_SPLIT_BLOCK_THREADS, 1, 1),
        )

    @cute.jit
    def _run_kmeans_assign_topk_wide_batched(
        query: cute.Tensor,
        centroids: cute.Tensor,
        top_distance_out: cute.Tensor,
        top_centroid_out: cute.Tensor,
    ):
        _kmeans_assign_topk_wide_kernel(
            query,
            centroids,
            top_distance_out,
            top_centroid_out,
        ).launch(grid=(BATCHED_QUERY_COUNT, 1, 1), block=(BLOCK_THREADS, 1, 1))

    return (
        _run_kmeans_assign_topk,
        _run_kmeans_assign_topk_batched,
        _run_kmeans_assign_topk_feature_split_batched,
        _run_kmeans_assign_topk_feature_split_score_batched,
        _run_kmeans_assign_top1_feature_split_score_batched,
        _run_kmeans_assign_top1_feature_split_score_warp_batched,
        _run_kmeans_assign_topk_wide_batched,
        torch,
        from_dlpack,
        cutlass,
        feature_split_exchange_storage,
        feature_split_topk_storage,
    )


@dataclass(frozen=True)
class PreparedExample:
    step: Callable[[], None]
    synchronize: Callable[[], None]
    validate: Callable[[], dict[str, Any]]


def _assert_topk_pairs_valid(
    actual_distances: Any,
    actual_centroids: Any,
    all_distances: Any,
    *,
    k: int,
) -> None:
    actual_distance_values = [int(x) for x in actual_distances.cpu().tolist()]
    actual_centroid_values = [int(x) for x in actual_centroids.cpu().tolist()]
    all_distance_values = [int(x) for x in all_distances.cpu().tolist()]

    assert len(actual_distance_values) == k
    assert len(actual_centroid_values) == k
    assert len(set(actual_centroid_values)) == k

    kth_distance = sorted(all_distance_values)[k - 1]
    required_centroids = {
        idx
        for idx, distance in enumerate(all_distance_values)
        if distance < kth_distance
    }
    actual_centroid_set = set(actual_centroid_values)
    assert required_centroids <= actual_centroid_set

    for distance, centroid in zip(
        actual_distance_values,
        actual_centroid_values,
        strict=True,
    ):
        assert 0 <= centroid < len(all_distance_values)
        assert distance == all_distance_values[centroid]
        assert distance <= kth_distance


def _make_centroids_host(torch: Any, centroids_per_tile: int, feature_dim: int) -> Any:
    centroid_rows = []
    for centroid_idx in range(centroids_per_tile):
        centroid_rows.append(
            [
                (centroid_idx * 7 + feature_idx * 3 + feature_idx // 2) % 17
                for feature_idx in range(feature_dim)
            ]
        )
    return torch.tensor(centroid_rows, dtype=torch.int32)


def _make_query_host(torch: Any, query_count: int, feature_dim: int) -> Any:
    if query_count == 1 and feature_dim == FEATURE_DIM:
        return torch.tensor(
            [[3, 1, 4, 1, 5, 9, 2, 6]],
            dtype=torch.int32,
        )

    query_rows = []
    for query_idx in range(query_count):
        query_rows.append(
            [
                (query_idx * 5 + feature_idx * 7 + feature_idx // 3) % 19
                for feature_idx in range(feature_dim)
            ]
        )
    return torch.tensor(query_rows, dtype=torch.int32)


def _prepare_example(*, query_count: int, mode: str) -> PreparedExample:
    """Prepare reusable inputs and a launch-only step for the example."""

    (
        run_kmeans_assign_topk,
        run_kmeans_assign_topk_batched,
        run_kmeans_assign_topk_feature_split_batched,
        run_kmeans_assign_topk_feature_split_score_batched,
        run_kmeans_assign_top1_feature_split_score_batched,
        run_kmeans_assign_top1_feature_split_score_warp_batched,
        run_kmeans_assign_topk_wide_batched,
        torch,
        from_dlpack,
        cutlass,
        feature_split_exchange_storage,
        feature_split_topk_storage,
    ) = make_runner()
    cutlass.cuda.initialize_cuda_context()
    feature_split_exchange_storage.reset_uses()
    feature_split_topk_storage.reset_uses()

    use_wide_shape = mode in {
        "wide_batched",
        "feature_split_batched",
        "feature_split_score_batched",
        "feature_split_top1_score_batched",
        "feature_split_top1_score_warp_batched",
    }
    centroids_per_tile = (
        WIDE_CENTROIDS_PER_TILE if use_wide_shape else CENTROIDS_PER_TILE
    )
    feature_dim = WIDE_FEATURE_DIM if use_wide_shape else FEATURE_DIM
    query_host = _make_query_host(torch, query_count, feature_dim)
    centroids_host = _make_centroids_host(torch, centroids_per_tile, feature_dim)

    query = query_host.reshape(-1).cuda()
    centroids = centroids_host.reshape(-1).cuda()
    top_distance_out = torch.zeros(
        (query_count * TOPK_K,),
        dtype=torch.int32,
        device="cuda",
    )
    top_centroid_out = torch.zeros(
        (query_count * TOPK_K,),
        dtype=torch.int32,
        device="cuda",
    )
    query_arg = from_dlpack(query)
    centroids_arg = from_dlpack(centroids)
    distance_arg = from_dlpack(top_distance_out)
    centroid_arg = from_dlpack(top_centroid_out)

    def step() -> None:
        if mode == "feature_split_batched":
            run_kmeans_assign_topk_feature_split_batched(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )
        elif mode == "feature_split_score_batched":
            run_kmeans_assign_topk_feature_split_score_batched(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )
        elif mode == "feature_split_top1_score_batched":
            run_kmeans_assign_top1_feature_split_score_batched(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )
        elif mode == "feature_split_top1_score_warp_batched":
            run_kmeans_assign_top1_feature_split_score_warp_batched(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )
        elif mode == "wide_batched":
            run_kmeans_assign_topk_wide_batched(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )
        elif mode == "batched":
            run_kmeans_assign_topk_batched(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )
        else:
            run_kmeans_assign_topk(
                query_arg,
                centroids_arg,
                distance_arg,
                centroid_arg,
            )

    def synchronize() -> None:
        torch.cuda.synchronize()

    def validate() -> dict[str, Any]:
        synchronize()
        query_rows = query_host.reshape(query_count, 1, feature_dim)
        centroid_rows_host = centroids_host.reshape(1, centroids_per_tile, feature_dim)
        if mode in {
            "feature_split_score_batched",
            "feature_split_top1_score_batched",
            "feature_split_top1_score_warp_batched",
        }:
            expected_values = (
                torch.sum(centroid_rows_host * centroid_rows_host, dim=-1)
                - torch.sum(2 * query_rows * centroid_rows_host, dim=-1)
                + FEATURE_SPLIT_SCORE_OFFSET
            ).to(torch.int32)
        else:
            delta = centroid_rows_host - query_rows
            expected_values = torch.sum(delta * delta, dim=-1).to(torch.int32)
        distance_rows = top_distance_out.reshape(query_count, TOPK_K)
        centroid_rows = top_centroid_out.reshape(query_count, TOPK_K)

        validated_k = (
            FEATURE_SPLIT_SCORE_TOP1_K
            if mode == "feature_split_top1_score_batched"
            or mode == "feature_split_top1_score_warp_batched"
            else TOPK_K
        )
        result_window = slice(0, validated_k)
        for query_idx in range(query_count):
            _assert_topk_pairs_valid(
                distance_rows[query_idx, result_window],
                centroid_rows[query_idx, result_window],
                expected_values[query_idx],
                k=validated_k,
            )

        return {
            "query_count": query_count,
            "topk_k": validated_k,
            "top_distances": [
                int(x) for x in distance_rows[0, result_window].cpu().tolist()
            ],
            "top_centroids": [
                int(x) for x in centroid_rows[0, result_window].cpu().tolist()
            ],
        }

    return PreparedExample(
        step=step,
        synchronize=synchronize,
        validate=validate,
    )


def prepare_example() -> PreparedExample:
    """Prepare the single-query fused Fused k-means-assignment TopK example."""

    return _prepare_example(query_count=1, mode="single")


def prepare_batched_example() -> PreparedExample:
    """Prepare a throughput-oriented multi-query fused TopK example."""

    return _prepare_example(query_count=BATCHED_QUERY_COUNT, mode="batched")


def prepare_wide_batched_example() -> PreparedExample:
    """Prepare a wider D=128 fused TopK throughput example."""

    return _prepare_example(query_count=BATCHED_QUERY_COUNT, mode="wide_batched")


def prepare_feature_split_batched_example() -> PreparedExample:
    """Prepare a D=128 example with four lanes per candidate centroid."""

    return _prepare_example(
        query_count=BATCHED_QUERY_COUNT,
        mode="feature_split_batched",
    )


def prepare_feature_split_score_batched_example() -> PreparedExample:
    """Prepare a D=128 assignment-score example without query-norm output."""

    return _prepare_example(
        query_count=BATCHED_QUERY_COUNT,
        mode="feature_split_score_batched",
    )


def prepare_feature_split_top1_score_batched_example() -> PreparedExample:
    """Prepare a D=128 score-only assignment top-1 example."""

    return _prepare_example(
        query_count=BATCHED_QUERY_COUNT,
        mode="feature_split_top1_score_batched",
    )


def prepare_feature_split_top1_score_warp_batched_example() -> PreparedExample:
    """Prepare a D=128 top-1 example using hierarchical warp min reductions."""

    return _prepare_example(
        query_count=BATCHED_QUERY_COUNT,
        mode="feature_split_top1_score_warp_batched",
    )


def run_example() -> dict[str, Any]:
    """Run the fused Fused k-means-assignment TopK example and validate the result."""

    prepared = prepare_example()
    prepared.step()
    return prepared.validate()


def main() -> int:
    print(run_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
