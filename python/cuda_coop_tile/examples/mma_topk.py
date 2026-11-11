# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""cuTile MMA, local row TopK, and candidate-merge TopK.

This fixed-shape example computes 16x128 output tiles for a matrix with 512
columns. Each MMA CTA keeps eight candidates per output row. A second kernel
selects the final eight from 4 * 8 candidates, without materializing the full
GEMM output.

Both kernels use four worker warps. The merge loads the 32 stored candidates
into a 128-element Tile with zero padding. Its runtime valid count comes from
the intermediate array width, so the padded entries are excluded from TopK.
"""

from __future__ import annotations

import cuda.tile as ct

import cuda.coop.tile as coop

M_TILE = 16
N_TILE = 128
K_TILE = 32
COLUMN_TILES = 4
TOPK = 8
CANDIDATES_PER_ROW = COLUMN_TILES * TOPK
MERGE_TILE_ITEMS = 128


@ct.kernel(num_worker_warps=4)
def mma_local_topk_kernel(a_array, b_array, partial_scores, partial_columns):
    tile_m = ct.bid(0)
    tile_n = ct.bid(1)
    accumulator = ct.full((M_TILE, N_TILE), 0.0, dtype=ct.float32)
    input_dtype = ct.tfloat32 if a_array.dtype == ct.float32 else a_array.dtype
    num_k_tiles = ct.num_tiles(a_array, axis=1, shape=(M_TILE, K_TILE))

    for tile_k in range(num_k_tiles):
        a_tile = ct.load(
            a_array,
            index=(tile_m, tile_k),
            shape=(M_TILE, K_TILE),
        ).astype(input_dtype)
        b_tile = ct.load(
            b_array,
            index=(tile_k, tile_n),
            shape=(K_TILE, N_TILE),
        ).astype(input_dtype)
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    global_columns = tile_n * N_TILE + ct.arange(N_TILE, dtype=ct.int32)
    for row_in_tile in range(M_TILE):
        score_row = ct.reshape(
            ct.extract(
                accumulator,
                index=(row_in_tile, 0),
                shape=(1, N_TILE),
            ),
            (N_TILE,),
        )
        top_scores, top_columns = coop.topk_max_pairs(
            score_row,
            global_columns,
            k=TOPK,
        )
        local_scores = ct.extract(top_scores, index=(0,), shape=(TOPK,))
        local_columns = ct.extract(top_columns, index=(0,), shape=(TOPK,))
        global_row = tile_m * M_TILE + row_in_tile
        ct.store(
            partial_scores,
            index=(global_row, tile_n),
            tile=ct.reshape(local_scores, (1, TOPK)),
        )
        ct.store(
            partial_columns,
            index=(global_row, tile_n),
            tile=ct.reshape(local_columns, (1, TOPK)),
        )


@ct.kernel(num_worker_warps=4)
def merge_row_candidates_kernel(
    partial_scores,
    partial_columns,
    top_scores_out,
    top_columns_out,
):
    row = ct.bid(0)
    candidate_scores = ct.reshape(
        ct.load(
            partial_scores,
            index=(row, 0),
            shape=(1, MERGE_TILE_ITEMS),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        (MERGE_TILE_ITEMS,),
    )
    candidate_columns = ct.reshape(
        ct.load(
            partial_columns,
            index=(row, 0),
            shape=(1, MERGE_TILE_ITEMS),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        (MERGE_TILE_ITEMS,),
    )
    top_scores, top_columns = coop.topk_max_pairs(
        candidate_scores,
        candidate_columns,
        k=TOPK,
        valid_items=partial_scores.shape[1],
    )
    final_scores = ct.extract(top_scores, index=(0,), shape=(TOPK,))
    final_columns = ct.extract(top_columns, index=(0,), shape=(TOPK,))
    ct.store(
        top_scores_out,
        index=(row, 0),
        tile=ct.reshape(final_scores, (1, TOPK)),
    )
    ct.store(
        top_columns_out,
        index=(row, 0),
        tile=ct.reshape(final_columns, (1, TOPK)),
    )


def launch_mma_topk(
    stream,
    a_array,
    b_array,
    partial_scores,
    partial_columns,
    top_scores_out,
    top_columns_out,
):
    """Launch the local-candidate producer and final row merge."""

    rows, inner = a_array.shape
    if rows % M_TILE != 0:
        raise ValueError(f"A rows must be divisible by {M_TILE}")
    if inner % K_TILE != 0:
        raise ValueError(f"A columns must be divisible by {K_TILE}")
    if b_array.shape != (inner, N_TILE * COLUMN_TILES):
        raise ValueError(
            f"B must have shape ({inner}, {N_TILE * COLUMN_TILES})"
        )
    if partial_scores.shape != (rows, CANDIDATES_PER_ROW):
        raise ValueError(
            f"partial scores must have shape ({rows}, {CANDIDATES_PER_ROW})"
        )
    if partial_columns.shape != partial_scores.shape:
        raise ValueError("partial score/column shapes must match")
    if top_scores_out.shape != (rows, TOPK):
        raise ValueError(f"top scores must have shape ({rows}, {TOPK})")
    if top_columns_out.shape != top_scores_out.shape:
        raise ValueError("top score/column shapes must match")

    ct.launch(
        stream,
        (rows // M_TILE, COLUMN_TILES, 1),
        mma_local_topk_kernel,
        (a_array, b_array, partial_scores, partial_columns),
    )
    ct.launch(
        stream,
        (rows, 1, 1),
        merge_row_candidates_kernel,
        (partial_scores, partial_columns, top_scores_out, top_columns_out),
    )
