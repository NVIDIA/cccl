# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import re
import shutil

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._group_rewrites import GroupRewriteError

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

THREADS = 32
MAPPED_GROUP_THREADS = 8


@cuda.jit
def _block_group_kernel(d_input, d_round_trip, d_rank, d_count, d_total):
    group = coop.this_block()
    warp = coop.this_warp()
    lanes = warp.group_by(MAPPED_GROUP_THREADS)
    warps = group.group_by(1)
    tid = cuda.threadIdx.x
    items = coop.ThreadData(1, dtype=types.int32)

    coop.load(group, d_input, items)
    warp.sync()
    lanes.sync_aligned()
    warps.sync()
    group.sync()
    coop.store(group, d_round_trip, items)

    d_rank[tid] = group.rank()
    d_count[tid] = group.count()
    d_total[tid] = coop.reduce(group, items[0])


@cuda.jit
def _mapped_warp_group_kernel(d_input, d_rank, d_count, d_member, d_total):
    tid = cuda.threadIdx.x
    group = coop.this_warp().group_by(MAPPED_GROUP_THREADS)

    d_rank[tid] = group.rank()
    d_count[tid] = group.count()
    d_member[tid] = group.is_member()
    d_total[tid] = coop.reduce(group, d_input[tid])


@cuda.jit
def _scalar_mapped_warp_group_kernel(d_input, d_total, lanes_per_group):
    tid = cuda.threadIdx.x
    group = coop.this_warp().group_by(lanes_per_group)
    d_total[tid] = coop.reduce(group, d_input[tid])


@cuda.jit
def _root_owned_partial_reduce_kernel(d_input, d_output, valid_items):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    total = coop.reduce(
        group,
        d_input[tid],
        broadcast=False,
        valid_items=valid_items,
    )
    if group.rank() == 0:
        d_output[0] = total


@cuda.jit
def _root_owned_partial_max_kernel(d_input, d_output, valid_items):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    total = coop.reduce(
        group,
        d_input[tid],
        binary_op="max",
        broadcast=False,
        valid_items=valid_items,
    )
    if group.rank() == 0:
        d_output[0] = total


@cuda.jit
def _scalar_store_kernel(d_input, d_block_output, d_warp_output):
    tid = cuda.threadIdx.x
    coop.store(coop.this_block(), d_block_output, d_input[tid])
    coop.store(coop.this_warp(), d_warp_output, d_input[tid])


@cuda.jit
def _literal_scalar_store_kernel(d_block_output, d_warp_output):
    coop.store(coop.this_block(), d_block_output, 7)
    coop.store(coop.this_warp(), d_warp_output, 7)


@cuda.jit
def _warp_load_store_x1_kernel(d_input, d_output, valid_items):
    group = coop.this_warp()
    items = coop.ThreadData(1, dtype=types.int32)
    coop.load(
        group,
        d_input,
        items,
        valid_items=valid_items,
        oob_default=np.int32(-7),
    )
    coop.store(group, d_output, items, valid_items=valid_items)


@cuda.jit
def _warp_load_store_x2_kernel(d_input, d_output, valid_items):
    group = coop.this_warp()
    items = coop.ThreadData(2, dtype=types.int32)
    coop.load(
        group,
        d_input,
        items,
        valid_items=valid_items,
        oob_default=np.int32(-7),
    )
    coop.store(group, d_output, items, valid_items=valid_items)


@cuda.jit
def _literal_reduce_kernel(d_output):
    d_output[cuda.threadIdx.x] = coop.reduce(coop.this_block(), 1)


@cuda.jit
def _inferred_thread_data_reduce_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(2)
    items[0] = d_input[2 * tid]
    items[1] = d_input[2 * tid + 1]
    d_output[tid] = coop.reduce(coop.this_block(), items)


@cuda.jit
def _descriptor_escape_kernel(d_output):
    group = coop.this_block()
    d_output[0] = 1 if group is None else 2


@cuda.jit
def _query_uint64_kernel(d_output):
    group = coop.this_block()
    d_output[cuda.threadIdx.x] = group.rank_as(types.uint64)


@cuda.jit
def _query_int16_kernel(d_output):
    group = coop.this_block()
    d_output[cuda.threadIdx.x] = group.rank_as(types.int16)


@cuda.jit
def _invalid_static_valid_items_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    d_output[tid] = coop.reduce(
        coop.this_block(),
        d_input[tid],
        broadcast=False,
        valid_items=0,
    )


@cuda.jit
def _bool_static_valid_items_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    d_output[tid] = coop.reduce(
        coop.this_block(),
        d_input[tid],
        broadcast=False,
        valid_items=True,
    )


@cuda.jit
def _grid_reduce_kernel(d_input, d_output):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    group = coop.this_grid()
    d_output[tid] = coop.reduce(group, d_input[tid])


@cuda.jit
def _grid_sync_kernel(d_input, d_output):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    group = coop.this_grid()
    group.sync()
    d_output[tid] = d_input[tid]


@cuda.jit
def _grid_query_kernel(d_rank, d_count):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    group = coop.this_grid()
    d_rank[tid] = group.rank()
    d_count[tid] = group.count()


@cuda.jit
def _cluster_group_kernel(d_input, d_rank, d_count, d_total):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    group = coop.this_cluster()
    group.sync()
    d_rank[tid] = group.rank()
    d_count[tid] = group.count()
    d_total[tid] = coop.reduce(group, d_input[tid])


@cuda.jit(device=True, inline="always")
def _device_group_sum(value):
    return coop.reduce(coop.this_block(), value)


@cuda.jit
def _device_helper_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    d_output[tid] = _device_group_sum(d_input[tid])


def test_block_group_queries_sync_load_store_and_broadcast_reduce():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_round_trip = np.zeros_like(h_input)
    h_rank = np.zeros_like(h_input)
    h_count = np.zeros_like(h_input)
    h_total = np.zeros_like(h_input)

    _block_group_kernel[1, THREADS](
        h_input,
        h_round_trip,
        h_rank,
        h_count,
        h_total,
    )

    np.testing.assert_array_equal(h_round_trip, h_input)
    np.testing.assert_array_equal(h_rank, np.arange(THREADS, dtype=np.int32))
    np.testing.assert_array_equal(h_count, np.full(THREADS, THREADS, dtype=np.int32))
    np.testing.assert_array_equal(
        h_total,
        np.full(THREADS, np.sum(h_input), dtype=np.int32),
    )


def test_mapped_warp_group_queries_membership_and_broadcast_reduce():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_rank = np.zeros_like(h_input)
    h_count = np.zeros_like(h_input)
    h_member = np.zeros_like(h_input)
    h_total = np.zeros_like(h_input)

    _mapped_warp_group_kernel[1, THREADS](
        h_input,
        h_rank,
        h_count,
        h_member,
        h_total,
    )

    expected_totals = np.repeat(
        h_input.reshape(-1, MAPPED_GROUP_THREADS).sum(axis=1),
        MAPPED_GROUP_THREADS,
    ).astype(np.int32)
    np.testing.assert_array_equal(
        h_rank,
        np.tile(
            np.arange(MAPPED_GROUP_THREADS, dtype=np.int32),
            THREADS // MAPPED_GROUP_THREADS,
        ),
    )
    np.testing.assert_array_equal(
        h_count,
        np.full(THREADS, MAPPED_GROUP_THREADS, dtype=np.int32),
    )
    np.testing.assert_array_equal(h_member, np.ones(THREADS, dtype=np.int32))
    np.testing.assert_array_equal(h_total, expected_totals)


# Regression: https://github.com/NVIDIA/numba-cuda-mlir/pull/239
@pytest.mark.parametrize("lanes_per_group", [8, 16])
def test_scalar_group_by_specializes_mapped_warp_reduce(lanes_per_group):
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_total = np.zeros_like(h_input)

    _scalar_mapped_warp_group_kernel[1, THREADS](
        h_input,
        h_total,
        lanes_per_group,
    )

    expected_totals = np.repeat(
        h_input.reshape(-1, lanes_per_group).sum(axis=1),
        lanes_per_group,
    ).astype(np.int32)
    np.testing.assert_array_equal(h_total, expected_totals)


def test_partial_reduce_routes_to_root_owned_cub_result():
    valid_items = 17
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros(1, dtype=np.int32)

    _root_owned_partial_reduce_kernel[1, THREADS](
        h_input,
        h_output,
        valid_items,
    )

    np.testing.assert_array_equal(
        h_output,
        np.asarray([np.sum(h_input[:valid_items])], dtype=np.int32),
    )

    h_output.fill(0)
    _root_owned_partial_max_kernel[1, THREADS](
        h_input,
        h_output,
        valid_items,
    )
    np.testing.assert_array_equal(
        h_output,
        np.asarray([np.max(h_input[:valid_items])], dtype=np.int32),
    )


def test_scalar_store_infers_dtype_for_block_and_each_physical_warp():
    threads = 2 * THREADS
    h_input = np.arange(1, threads + 1, dtype=np.int32)
    h_block_output = np.full_like(h_input, -1)
    h_warp_output = np.full_like(h_input, -1)

    _scalar_store_kernel[1, threads](
        h_input,
        h_block_output,
        h_warp_output,
    )

    np.testing.assert_array_equal(h_block_output, h_input)
    np.testing.assert_array_equal(h_warp_output, h_input)

    _literal_scalar_store_kernel[1, threads](h_block_output, h_warp_output)
    expected = np.full(threads, 7, dtype=np.int32)
    np.testing.assert_array_equal(h_block_output, expected)
    np.testing.assert_array_equal(h_warp_output, expected)


@pytest.mark.parametrize(
    ("kernel", "items_per_thread"),
    [
        (_warp_load_store_x1_kernel, 1),
        (_warp_load_store_x2_kernel, 2),
    ],
)
def test_each_physical_warp_owns_a_distinct_load_store_tile(kernel, items_per_thread):
    threads = 2 * THREADS
    tile_items = THREADS * items_per_thread
    valid_items = tile_items - 5
    h_input = np.arange(threads * items_per_thread, dtype=np.int32)
    h_output = np.full_like(h_input, -1)

    kernel[1, threads](
        h_input,
        h_output,
        valid_items,
    )

    expected = np.full_like(h_input, -1)
    expected[:valid_items] = h_input[:valid_items]
    expected[tile_items : tile_items + valid_items] = h_input[
        tile_items : tile_items + valid_items
    ]
    np.testing.assert_array_equal(h_output, expected)


def test_literal_and_inferred_thread_data_reduce_dtype():
    h_literal = np.zeros(THREADS, dtype=np.int64)
    _literal_reduce_kernel[1, THREADS](h_literal)
    np.testing.assert_array_equal(
        h_literal,
        np.full(THREADS, THREADS, dtype=np.int64),
    )

    h_input = np.arange(1, 2 * THREADS + 1, dtype=np.int32)
    h_output = np.zeros(THREADS, dtype=np.int32)
    _inferred_thread_data_reduce_kernel[1, THREADS](h_input, h_output)
    np.testing.assert_array_equal(
        h_output,
        np.full(THREADS, np.sum(h_input), dtype=np.int32),
    )


def test_descriptor_runtime_escape_fails_instead_of_becoming_none():
    h_output = np.zeros(1, dtype=np.int32)

    with pytest.raises(GroupRewriteError, match="would escape to runtime"):
        _descriptor_escape_kernel[1, THREADS](h_output)


def test_query_dtype_domain_matches_cutlass():
    h_output = np.zeros(THREADS, dtype=np.uint64)
    _query_uint64_kernel[1, THREADS](h_output)
    np.testing.assert_array_equal(
        h_output,
        np.arange(THREADS, dtype=np.uint64),
    )

    with pytest.raises(TypeError, match="query dtype must be one of"):
        _query_int16_kernel[1, THREADS](np.zeros(THREADS, dtype=np.int16))


@pytest.mark.parametrize(
    ("kernel", "exception", "message"),
    [
        (
            _invalid_static_valid_items_kernel,
            ValueError,
            "must be between 1 and group size",
        ),
        (
            _bool_static_valid_items_kernel,
            TypeError,
            "must be an integer, not bool",
        ),
    ],
)
def test_static_partial_reduce_count_is_checked(kernel, exception, message):
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    with pytest.raises(exception, match=message):
        kernel[1, THREADS](h_input, h_output)


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        (
            _grid_reduce_kernel,
            "grid groups require a hidden per-launch provider workspace",
        ),
        (
            _grid_sync_kernel,
            "grid synchronization requires a verified cooperative launch",
        ),
    ],
)
def test_unsupported_grid_operations_fail_clearly(kernel, message):
    h_input = np.arange(2 * THREADS, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    with pytest.raises(NotImplementedError, match=message):
        kernel[2, THREADS](h_input, h_output)


def test_grid_queries_use_exact_configured_launch_shape():
    members = 2 * THREADS
    h_rank = np.zeros(members, dtype=np.int32)
    h_count = np.zeros_like(h_rank)

    _grid_query_kernel[2, THREADS](h_rank, h_count)

    np.testing.assert_array_equal(h_rank, np.arange(members, dtype=np.int32))
    np.testing.assert_array_equal(h_count, np.full(members, members, dtype=np.int32))


def test_cluster_group_queries_sync_and_broadcast_reduce():
    if cuda.get_current_device().compute_capability < (9, 0):
        pytest.skip("thread-block clusters require compute capability 9.0 or newer")

    members = 2 * THREADS
    h_input = np.arange(1, members + 1, dtype=np.int32)
    h_rank = np.zeros_like(h_input)
    h_count = np.zeros_like(h_input)
    h_total = np.zeros_like(h_input)

    _cluster_group_kernel[2, THREADS, 0, 0, 2](
        h_input,
        h_rank,
        h_count,
        h_total,
    )

    np.testing.assert_array_equal(h_rank, np.arange(members, dtype=np.int32))
    np.testing.assert_array_equal(h_count, np.full(members, members, dtype=np.int32))
    np.testing.assert_array_equal(
        h_total,
        np.full(members, np.sum(h_input), dtype=np.int32),
    )


def test_device_helper_is_planned_after_inlining_and_has_no_call_frame_sass():
    h_input = np.arange(1, THREADS + 1, dtype=np.int32)
    h_output = np.zeros_like(h_input)

    _device_helper_kernel[1, THREADS](h_input, h_output)

    np.testing.assert_array_equal(
        h_output,
        np.full(THREADS, np.sum(h_input), dtype=np.int32),
    )
    if shutil.which("nvdisasm") is None:
        return

    sass = _device_helper_kernel.inspect_sass(_device_helper_kernel.signatures[0])
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None
