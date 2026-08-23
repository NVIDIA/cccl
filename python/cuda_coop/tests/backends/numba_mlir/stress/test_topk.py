# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop

from ..support.runtime import (
    ITEMS_PER_THREAD,
    THREADS,
    _assert_topk_stress_output,
    _make_topk_rank_flags,
    _make_topk_stress_inputs,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _block_topk_pair_stress_grid_stride_kernel(
    d_keys,
    d_values,
    d_rank_flags,
    d_keys_out,
    d_values_out,
    d_ranks_out,
    d_checksums,
    total_items,
    runtime_k,
):
    items_per_block = THREADS * ITEMS_PER_THREAD
    block_offset = cuda.blockIdx.x * items_per_block
    grid_stride = cuda.gridDim.x * items_per_block

    while block_offset < total_items:
        valid = total_items - block_offset
        if valid > items_per_block:
            valid = items_per_block
        if valid < 0:
            valid = 0
        valid_i32 = types.int32(valid)
        actual_k = types.int32(runtime_k)
        if actual_k > valid_i32:
            actual_k = valid_i32

        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
        values = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_values.dtype)
        flags = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_rank_flags.dtype)
        ranks = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_ranks_out.dtype)

        coop._block.load(
            d_keys,
            keys,
            valid_i32,
            types.int32(0),
            offset=block_offset,
        )
        coop._block.load(
            d_values,
            values,
            valid_i32,
            types.int32(0),
            offset=block_offset,
        )
        coop._block.topk_max_pairs(keys, values, actual_k, num_valid=valid_i32)
        coop._block.load(d_rank_flags, flags, offset=block_offset)

        local_checksum = types.int64(0)
        for i in range(ITEMS_PER_THREAD):
            linear = cuda.threadIdx.x * ITEMS_PER_THREAD + i
            if linear < actual_k:
                local_checksum = local_checksum + (
                    types.int64(keys[i]) * types.int64(1315423911)
                    + types.int64(values[i]) * types.int64(2654435761)
                )

        coop._block.scan(flags, ranks)
        block_checksum = coop._block.sum(
            local_checksum,
            dtype="int64",
            threads_per_block=THREADS,
        )

        tile_idx = block_offset // items_per_block
        if cuda.threadIdx.x == 0:
            d_checksums[tile_idx] = block_checksum

        coop._block.store(d_keys_out, keys, offset=block_offset)
        coop._block.store(d_values_out, values, offset=block_offset)
        coop._block.store(d_ranks_out, ranks, offset=block_offset)
        block_offset += grid_stride


@cuda.jit
def _block_topk_pair_stress_shared_storage_kernel(
    d_keys,
    d_values,
    d_rank_flags,
    d_keys_out,
    d_values_out,
    d_ranks_out,
    d_checksums,
    total_items,
    runtime_k,
):
    temp_storage = coop.TempStorage(sharing="shared")
    items_per_block = THREADS * ITEMS_PER_THREAD
    block_offset = cuda.blockIdx.x * items_per_block
    grid_stride = cuda.gridDim.x * items_per_block

    while block_offset < total_items:
        valid = total_items - block_offset
        if valid > items_per_block:
            valid = items_per_block
        if valid < 0:
            valid = 0
        valid_i32 = types.int32(valid)
        actual_k = types.int32(runtime_k)
        if actual_k > valid_i32:
            actual_k = valid_i32

        keys = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_keys.dtype)
        values = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_values.dtype)
        flags = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_rank_flags.dtype)
        ranks = coop.ThreadData(ITEMS_PER_THREAD, dtype=d_ranks_out.dtype)

        coop._block.load(
            d_keys,
            keys,
            valid_i32,
            types.int32(0),
            offset=block_offset,
            temp_storage=temp_storage,
        )
        coop._block.load(
            d_values,
            values,
            valid_i32,
            types.int32(0),
            offset=block_offset,
            temp_storage=temp_storage,
        )
        coop._block.topk_max_pairs(
            keys,
            values,
            actual_k,
            num_valid=valid_i32,
            temp_storage=temp_storage,
        )
        coop._block.load(
            d_rank_flags,
            flags,
            offset=block_offset,
            temp_storage=temp_storage,
        )

        local_checksum = types.int64(0)
        for i in range(ITEMS_PER_THREAD):
            linear = cuda.threadIdx.x * ITEMS_PER_THREAD + i
            if linear < actual_k:
                local_checksum = local_checksum + (
                    types.int64(keys[i]) * types.int64(1315423911)
                    + types.int64(values[i]) * types.int64(2654435761)
                )

        coop._block.scan(flags, ranks, temp_storage=temp_storage)
        block_checksum = coop._block.sum(
            local_checksum,
            dtype="int64",
            threads_per_block=THREADS,
            temp_storage=temp_storage,
        )

        tile_idx = block_offset // items_per_block
        if cuda.threadIdx.x == 0:
            d_checksums[tile_idx] = block_checksum

        coop._block.store(
            d_keys_out, keys, offset=block_offset, temp_storage=temp_storage
        )
        coop._block.store(
            d_values_out,
            values,
            offset=block_offset,
            temp_storage=temp_storage,
        )
        coop._block.store(
            d_ranks_out, ranks, offset=block_offset, temp_storage=temp_storage
        )
        block_offset += grid_stride


def test_block_topk_pair_stress_grid_stride_and_shared_storage():
    tile_size = THREADS * ITEMS_PER_THREAD
    num_tiles = 4
    total_items = tile_size * (num_tiles - 1) + 47
    k = 17
    launch_blocks = 2

    h_keys, h_values = _make_topk_stress_inputs(tile_size, num_tiles, total_items)
    h_rank_flags = _make_topk_rank_flags(tile_size, num_tiles, total_items, k)
    padded_items = num_tiles * tile_size

    for kernel in (
        _block_topk_pair_stress_grid_stride_kernel,
        _block_topk_pair_stress_shared_storage_kernel,
    ):
        h_keys_out = np.zeros(padded_items, dtype=np.int32)
        h_values_out = np.zeros(padded_items, dtype=np.int32)
        h_ranks_out = np.zeros(padded_items, dtype=np.int32)
        h_checksums = np.zeros(num_tiles, dtype=np.int64)

        kernel[launch_blocks, THREADS](
            h_keys,
            h_values,
            h_rank_flags,
            h_keys_out,
            h_values_out,
            h_ranks_out,
            h_checksums,
            np.int32(total_items),
            np.int32(k),
        )

        _assert_topk_stress_output(
            h_keys,
            h_values,
            h_keys_out,
            h_values_out,
            h_ranks_out,
            h_checksums,
            tile_size,
            num_tiles,
            total_items,
            k,
        )
