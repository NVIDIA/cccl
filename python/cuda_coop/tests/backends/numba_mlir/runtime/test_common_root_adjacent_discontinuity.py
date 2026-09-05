# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD
_COMMON_SEGMENTS = 5
_QUALIFIED_SEGMENTS = 7

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_adjacent_discontinuity_kernel(
    source,
    output,
    valid_items,
    predecessor,
    successor,
):
    tid = cuda.threadIdx.x
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = coop.this_block()
    storage = coop.TempStorage()
    left = coop.adjacent_difference(
        group,
        items,
        valid_items=valid_items,
        tile_predecessor_item=predecessor,
        temp_storage=storage,
    )
    right = coop.adjacent_difference(
        group,
        items,
        direction="right",
        tile_successor_item=successor,
        temp_storage=storage,
    )
    heads = coop.discontinuity(
        group,
        items,
        mode="heads",
        tile_predecessor_item=predecessor,
        temp_storage=storage,
    )
    tails = coop.discontinuity(
        group,
        items,
        mode="tails",
        tile_successor_item=successor,
        temp_storage=storage,
    )

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        output[0 * _TILE_ITEMS + item] = items[index]
        output[1 * _TILE_ITEMS + item] = left[index]
        output[2 * _TILE_ITEMS + item] = right[index]
        output[3 * _TILE_ITEMS + item] = heads[index]
        output[4 * _TILE_ITEMS + item] = tails[index]


@cuda.jit
def _qualified_adjacent_discontinuity_kernel(
    source,
    output,
    valid_items,
    predecessor,
    successor,
):
    tid = cuda.threadIdx.x
    items = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = numba_coop.this_block()
    storage = numba_coop.TempStorage()
    left = numba_coop.adjacent_difference(
        group,
        items,
        valid_items=valid_items,
        tile_predecessor_item=predecessor,
        temp_storage=storage,
    )
    right = numba_coop.adjacent_difference(
        group,
        items,
        direction="right",
        tile_successor_item=successor,
        temp_storage=storage,
    )
    heads = numba_coop.discontinuity(
        group,
        items,
        mode="heads",
        tile_predecessor_item=predecessor,
        temp_storage=storage,
    )
    tails = numba_coop.discontinuity(
        group,
        items,
        mode="tails",
        tile_successor_item=successor,
        temp_storage=storage,
    )
    pair_heads, pair_tails = numba_coop.discontinuity(
        group,
        items,
        mode="heads_and_tails",
        tile_predecessor_item=predecessor,
        tile_successor_item=successor,
        temp_storage=storage,
    )

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        output[0 * _TILE_ITEMS + item] = items[index]
        output[1 * _TILE_ITEMS + item] = left[index]
        output[2 * _TILE_ITEMS + item] = right[index]
        output[3 * _TILE_ITEMS + item] = heads[index]
        output[4 * _TILE_ITEMS + item] = tails[index]
        output[5 * _TILE_ITEMS + item] = pair_heads[index]
        output[6 * _TILE_ITEMS + item] = pair_tails[index]


def _independent_oracle(values, valid_items, predecessor, successor):
    expected = np.empty((_QUALIFIED_SEGMENTS, _TILE_ITEMS), dtype=np.int32)
    expected[0] = values

    expected[1] = values
    expected[1, 0] = values[0] - predecessor
    expected[1, 1:valid_items] = values[1:valid_items] - values[: valid_items - 1]

    expected[2, :-1] = values[:-1] - values[1:]
    expected[2, -1] = values[-1] - successor

    expected[3, 0] = values[0] != predecessor
    expected[3, 1:] = values[1:] != values[:-1]
    expected[4, :-1] = values[:-1] != values[1:]
    expected[4, -1] = values[-1] != successor
    expected[5] = expected[3]
    expected[6] = expected[4]
    return expected


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.discontinuity", backend="numba_mlir", evidence="runtime"
)
def test_common_adjacent_discontinuity_matches_qualified_and_independent_oracle():
    indices = np.arange(_TILE_ITEMS, dtype=np.int32)
    values = ((indices * 11 + indices // 3) % 17) - 8
    original = values.copy()
    valid_items = np.int32(_TILE_ITEMS - 3)
    predecessor = np.int32(-13)
    successor = np.int32(29)
    common_output = np.full(
        _COMMON_SEGMENTS * _TILE_ITEMS,
        -999,
        dtype=np.int32,
    )
    qualified_output = np.full(
        _QUALIFIED_SEGMENTS * _TILE_ITEMS,
        -999,
        dtype=np.int32,
    )

    _common_adjacent_discontinuity_kernel[1, _THREADS](
        values,
        common_output,
        valid_items,
        predecessor,
        successor,
    )
    _qualified_adjacent_discontinuity_kernel[1, _THREADS](
        values,
        qualified_output,
        valid_items,
        predecessor,
        successor,
    )
    cuda.synchronize()

    expected = _independent_oracle(
        original,
        int(valid_items),
        predecessor,
        successor,
    )
    common_segments = common_output.reshape(_COMMON_SEGMENTS, _TILE_ITEMS)
    qualified_segments = qualified_output.reshape(
        _QUALIFIED_SEGMENTS,
        _TILE_ITEMS,
    )
    assert common_output.dtype == np.int32
    assert qualified_output.dtype == np.int32
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(common_segments[0], original)
    np.testing.assert_array_equal(qualified_segments[0], original)
    np.testing.assert_array_equal(
        common_segments,
        qualified_segments[:_COMMON_SEGMENTS],
    )
    np.testing.assert_array_equal(common_segments, expected[:_COMMON_SEGMENTS])
    np.testing.assert_array_equal(qualified_segments, expected)


@cuda.jit(device=True)
def _complex_subtract(left, right):
    return left - right


@cuda.jit(device=True)
def _complex_not_equal(left, right):
    return left.real != right.real or left.imag != right.imag


@cuda.jit
def _qualified_complex_adjacent_discontinuity_kernel(
    source,
    left_output,
    right_output,
    head_output,
    tail_output,
    pair_head_output,
    pair_tail_output,
    valid_items,
    predecessor,
    successor,
):
    tid = cuda.threadIdx.x
    items = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.complex128)
    for index in range(_ITEMS_PER_THREAD):
        items[index] = source[tid * _ITEMS_PER_THREAD + index]

    group = numba_coop.this_block()
    storage = numba_coop.TempStorage()
    left = numba_coop.adjacent_difference(
        group,
        items,
        valid_items=valid_items,
        tile_predecessor_item=predecessor,
        temp_storage=storage,
        difference_op=_complex_subtract,
    )
    right = numba_coop.adjacent_difference(
        group,
        items,
        direction="right",
        tile_successor_item=successor,
        temp_storage=storage,
        difference_op=_complex_subtract,
    )
    heads = numba_coop.discontinuity(
        group,
        items,
        mode="heads",
        tile_predecessor_item=predecessor,
        temp_storage=storage,
        flag_op=_complex_not_equal,
    )
    tails = numba_coop.discontinuity(
        group,
        items,
        mode="tails",
        tile_successor_item=successor,
        temp_storage=storage,
        flag_op=_complex_not_equal,
    )
    pair_heads, pair_tails = numba_coop.discontinuity(
        group,
        items,
        mode="heads_and_tails",
        tile_predecessor_item=predecessor,
        tile_successor_item=successor,
        temp_storage=storage,
        flag_op=_complex_not_equal,
    )

    for index in range(_ITEMS_PER_THREAD):
        item = tid * _ITEMS_PER_THREAD + index
        left_output[item] = left[index]
        right_output[item] = right[index]
        head_output[item] = heads[index]
        tail_output[item] = tails[index]
        pair_head_output[item] = pair_heads[index]
        pair_tail_output[item] = pair_tails[index]


def test_qualified_segmentation_supports_value_level_complex128_callbacks():
    indices = np.arange(_TILE_ITEMS, dtype=np.int32)
    values = (
        (indices // 3).astype(np.float64) + 1j * ((indices * 2) % 7).astype(np.float64)
    ).astype(np.complex128)
    original = values.copy()
    valid_items = np.int32(_TILE_ITEMS - 3)
    predecessor = np.complex128(-13.0 + 7.0j)
    successor = np.complex128(29.0 - 11.0j)
    complex_sentinel = np.complex128(-999.0 - 777.0j)
    flag_sentinel = np.int32(-99)
    left = np.full_like(values, complex_sentinel)
    right = np.full_like(values, complex_sentinel)
    flag_outputs = [
        np.full(_TILE_ITEMS, flag_sentinel, dtype=np.int32) for _ in range(4)
    ]

    _qualified_complex_adjacent_discontinuity_kernel[1, _THREADS](
        values,
        left,
        right,
        *flag_outputs,
        valid_items,
        predecessor,
        successor,
    )
    cuda.synchronize()

    expected_left = values.copy()
    expected_left[0] = values[0] - predecessor
    expected_left[1:valid_items] = values[1:valid_items] - values[: valid_items - 1]
    expected_right = np.empty_like(values)
    expected_right[:-1] = values[:-1] - values[1:]
    expected_right[-1] = values[-1] - successor
    expected_heads = np.empty(_TILE_ITEMS, dtype=np.int32)
    expected_heads[0] = values[0] != predecessor
    expected_heads[1:] = values[1:] != values[:-1]
    expected_tails = np.empty(_TILE_ITEMS, dtype=np.int32)
    expected_tails[:-1] = values[:-1] != values[1:]
    expected_tails[-1] = values[-1] != successor

    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(left[:valid_items], expected_left[:valid_items])
    np.testing.assert_array_equal(right, expected_right)
    np.testing.assert_array_equal(flag_outputs[0], expected_heads)
    np.testing.assert_array_equal(flag_outputs[1], expected_tails)
    np.testing.assert_array_equal(flag_outputs[2], expected_heads)
    np.testing.assert_array_equal(flag_outputs[3], expected_tails)
