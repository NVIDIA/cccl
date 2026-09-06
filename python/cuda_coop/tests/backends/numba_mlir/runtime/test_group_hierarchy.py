# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Runtime qualification for Numba-CUDA-MLIR group hierarchy methods."""

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as portable_coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_BLOCKS = 2
_BLOCK_THREADS = 96
_THREADS = _BLOCKS * _BLOCK_THREADS
_QUERY_FIELDS = 19


@cuda.jit
def _group_query_kernel(output):
    block_thread = cuda.threadIdx.x
    thread = portable_coop.this_thread()
    warp = qualified_coop.this_warp()
    block = portable_coop.this_block()
    lanes = qualified_coop.this_warp().group_by(8)
    warps = portable_coop.this_block().group_by(2, exhaustive=False)

    thread.sync()
    warp.sync_aligned()
    lanes.sync()
    lanes.sync_aligned()
    block.sync()

    thread_index = cuda.blockIdx.x * _BLOCK_THREADS + block_thread
    output[0 * _THREADS + thread_index] = thread.rank("block")
    output[1 * _THREADS + thread_index] = thread.count("thread")
    output[2 * _THREADS + thread_index] = warp.rank("thread")
    output[3 * _THREADS + thread_index] = warp.count("block")
    output[4 * _THREADS + thread_index] = block.rank("thread")
    output[5 * _THREADS + thread_index] = block.count("grid")
    output[6 * _THREADS + thread_index] = lanes.rank("thread")
    output[7 * _THREADS + thread_index] = lanes.count("warp")
    mapped_member = warps.is_member()
    if mapped_member:
        output[8 * _THREADS + thread_index] = warps.rank("warp")
    else:
        output[8 * _THREADS + thread_index] = -1
    output[9 * _THREADS + thread_index] = warps.count("thread")
    output[10 * _THREADS + thread_index] = thread.is_member()
    output[11 * _THREADS + thread_index] = warp.is_member()
    output[12 * _THREADS + thread_index] = block.is_member()
    output[13 * _THREADS + thread_index] = lanes.is_member()
    output[14 * _THREADS + thread_index] = mapped_member
    if mapped_member:
        output[15 * _THREADS + thread_index] = warps.rank("block")
        output[16 * _THREADS + thread_index] = warps.count("block")
    else:
        output[15 * _THREADS + thread_index] = -1
        output[16 * _THREADS + thread_index] = -1
    output[17 * _THREADS + thread_index] = block.rank_as(
        types.uint64,
        "thread",
    )
    output[18 * _THREADS + thread_index] = thread.count_as(types.int16)


def test_physical_and_mapped_group_queries_match_independent_oracles() -> None:
    output = np.full(_QUERY_FIELDS * _THREADS, -99, dtype=np.int64)

    _group_query_kernel[_BLOCKS, _BLOCK_THREADS](output)

    block_thread = np.tile(np.arange(_BLOCK_THREADS, dtype=np.int64), _BLOCKS)
    warp_rank = block_thread // 32
    lane_rank = block_thread % 32
    mapped_member = warp_rank < 2
    expected = np.stack(
        (
            block_thread,
            np.ones(_THREADS, dtype=np.int64),
            lane_rank,
            np.full(_THREADS, 3, dtype=np.int64),
            block_thread,
            np.full(_THREADS, _BLOCKS, dtype=np.int64),
            lane_rank % 8,
            np.full(_THREADS, 4, dtype=np.int64),
            np.where(mapped_member, warp_rank % 2, -1),
            np.full(_THREADS, 64, dtype=np.int64),
            np.ones(_THREADS, dtype=np.int64),
            np.ones(_THREADS, dtype=np.int64),
            np.ones(_THREADS, dtype=np.int64),
            np.ones(_THREADS, dtype=np.int64),
            mapped_member.astype(np.int64),
            np.where(mapped_member, warp_rank // 2, -1),
            np.where(mapped_member, 1, -1),
            block_thread,
            np.ones(_THREADS, dtype=np.int64),
        )
    )
    np.testing.assert_array_equal(
        output.reshape(_QUERY_FIELDS, _THREADS),
        expected,
    )


@cuda.jit
def _partial_warp_query_kernel(observed_rank, observed_count):
    block = portable_coop.this_block()
    observed_rank[cuda.threadIdx.x] = block.rank("warp")
    observed_count[cuda.threadIdx.x] = block.count("warp")


def test_block_warp_queries_include_a_partial_final_warp() -> None:
    block_threads = 48
    observed_rank = np.full(block_threads, np.iinfo(np.uint32).max, dtype=np.uint32)
    observed_count = np.zeros(block_threads, dtype=np.uint32)

    _partial_warp_query_kernel[1, block_threads](observed_rank, observed_count)

    expected_rank = np.arange(block_threads, dtype=np.uint32) // 32
    np.testing.assert_array_equal(observed_rank, expected_rank)
    np.testing.assert_array_equal(
        observed_count,
        np.full(block_threads, 2, dtype=np.uint32),
    )
