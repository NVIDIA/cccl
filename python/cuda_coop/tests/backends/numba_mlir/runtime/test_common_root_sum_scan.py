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
_VALID_BLOCK_ITEMS = _THREADS - 5
_VALID_WARP_ITEMS = _THREADS - 5

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _common_root_block_sum_scan_kernel(
    d_input,
    d_sum,
    d_partial_sum,
    d_scan,
    d_exclusive_sum,
    d_inclusive_sum,
    d_exclusive_scan,
    d_inclusive_scan,
    d_original,
):
    tid = cuda.threadIdx.x
    group = coop.this_block()
    storage = coop.TempStorage()
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        items[item] = d_input[tid * _ITEMS_PER_THREAD + item]

    d_sum[tid] = coop.sum(group, items)
    partial_sum = coop.sum(
        group,
        d_input[tid * _ITEMS_PER_THREAD],
        broadcast=False,
        valid_items=_VALID_BLOCK_ITEMS,
        algorithm="raking",
    )
    if tid == 0:
        d_partial_sum[0] = partial_sum
    scanned = coop.scan(
        group,
        items,
        mode="inclusive",
        scan_op="max",
        algorithm="warp_scans",
        temp_storage=storage,
    )
    exclusive_sum = coop.exclusive_sum(group, items, temp_storage=storage)
    inclusive_sum = coop.inclusive_sum(group, items, temp_storage=storage)
    exclusive_scan = coop.exclusive_scan(
        group,
        items,
        scan_op="max",
        initial_value=0,
        temp_storage=storage,
    )
    inclusive_scan = coop.inclusive_scan(
        group,
        items,
        scan_op="max",
        temp_storage=storage,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        d_scan[index] = scanned[item]
        d_exclusive_sum[index] = exclusive_sum[item]
        d_inclusive_sum[index] = inclusive_sum[item]
        d_exclusive_scan[index] = exclusive_scan[item]
        d_inclusive_scan[index] = inclusive_scan[item]
        d_original[index] = items[item]


@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="runtime")
@pytest.mark.evidence_for("group.scan", backend="numba_mlir", evidence="runtime")
@pytest.mark.evidence_for(
    "group.exclusive_sum", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.inclusive_sum", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.exclusive_scan", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.inclusive_scan", backend="numba_mlir", evidence="runtime"
)
def test_common_root_block_sum_scan_preserves_shape_and_input(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    values = np.arange(1, _TILE_ITEMS + 1, dtype=np.int32)
    scalar_outputs = [
        np.zeros(_THREADS, dtype=np.int32),
        np.full(1, -999, dtype=np.int32),
    ]
    payload_outputs = [np.zeros_like(values) for _ in range(6)]

    _common_root_block_sum_scan_kernel[1, _THREADS](
        values,
        *scalar_outputs,
        *payload_outputs,
    )
    cuda.synchronize()

    np.testing.assert_array_equal(
        scalar_outputs[0],
        np.full(_THREADS, values.sum(), dtype=np.int32),
    )
    partial_input = values[::_ITEMS_PER_THREAD][:_VALID_BLOCK_ITEMS]
    np.testing.assert_array_equal(
        scalar_outputs[1],
        np.asarray([partial_input.sum()], dtype=np.int32),
    )
    expected_exclusive_sum = np.concatenate(
        (np.asarray([0], dtype=np.int32), np.cumsum(values[:-1], dtype=np.int32))
    )
    expected_exclusive_max = np.concatenate(
        (np.asarray([0], dtype=np.int32), np.maximum.accumulate(values[:-1]))
    )
    for actual, expected in zip(
        payload_outputs,
        (
            np.maximum.accumulate(values),
            expected_exclusive_sum,
            np.cumsum(values, dtype=np.int32),
            expected_exclusive_max,
            np.maximum.accumulate(values),
            values,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@cuda.jit
def _common_root_warp_sum_scan_kernel(
    d_input,
    d_sum,
    d_partial_sum,
    d_scan,
    d_exclusive_sum,
    d_inclusive_sum,
    d_exclusive_scan,
    d_inclusive_scan,
):
    tid = cuda.threadIdx.x
    group = coop.this_warp()
    value = d_input[tid]
    d_sum[tid] = coop.sum(group, value)
    partial_sum = coop.sum(
        group,
        value,
        broadcast=False,
        valid_items=_VALID_WARP_ITEMS,
    )
    if tid % _THREADS == 0:
        d_partial_sum[tid // _THREADS] = partial_sum
    d_scan[tid] = coop.scan(group, value, mode="inclusive", scan_op="max")
    d_exclusive_sum[tid] = coop.exclusive_sum(group, value)
    d_inclusive_sum[tid] = coop.inclusive_sum(group, value)
    d_exclusive_scan[tid] = coop.exclusive_scan(
        group,
        value,
        scan_op="max",
        initial_value=0,
    )
    d_inclusive_scan[tid] = coop.inclusive_scan(group, value, scan_op="max")


@pytest.mark.evidence_for("group.sum", backend="numba_mlir", evidence="runtime")
@pytest.mark.evidence_for("group.scan", backend="numba_mlir", evidence="runtime")
@pytest.mark.evidence_for(
    "group.exclusive_sum", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.inclusive_sum", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.exclusive_scan", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.inclusive_scan", backend="numba_mlir", evidence="runtime"
)
def test_common_root_physical_warps_are_independent(numba_mlir_cuda_available):
    del numba_mlir_cuda_available
    warps = 2
    threads = warps * _THREADS
    values = np.arange(1, threads + 1, dtype=np.int32)
    outputs = [np.zeros_like(values), np.full(warps, -999, dtype=np.int32)]
    outputs.extend(np.zeros_like(values) for _ in range(5))

    _common_root_warp_sum_scan_kernel[1, threads](values, *outputs)
    cuda.synchronize()

    expected_sum = np.empty_like(values)
    expected_partial_sum = np.empty(warps, dtype=np.int32)
    expected_scan = np.empty_like(values)
    expected_exclusive_sum = np.empty_like(values)
    expected_inclusive_sum = np.empty_like(values)
    expected_exclusive_scan = np.empty_like(values)
    expected_inclusive_scan = np.empty_like(values)
    for warp in range(warps):
        begin = warp * _THREADS
        end = begin + _THREADS
        warp_values = values[begin:end]
        expected_sum[begin:end] = warp_values.sum()
        expected_partial_sum[warp] = warp_values[:_VALID_WARP_ITEMS].sum()
        expected_scan[begin:end] = np.maximum.accumulate(warp_values)
        expected_exclusive_sum[begin] = 0
        expected_exclusive_sum[begin + 1 : end] = np.cumsum(
            warp_values[:-1], dtype=np.int32
        )
        expected_inclusive_sum[begin:end] = np.cumsum(warp_values, dtype=np.int32)
        expected_exclusive_scan[begin] = 0
        expected_exclusive_scan[begin + 1 : end] = np.maximum.accumulate(
            warp_values[:-1]
        )
        expected_inclusive_scan[begin:end] = np.maximum.accumulate(warp_values)

    for actual, expected in zip(
        outputs,
        (
            expected_sum,
            expected_partial_sum,
            expected_scan,
            expected_exclusive_sum,
            expected_inclusive_sum,
            expected_exclusive_scan,
            expected_inclusive_scan,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@cuda.jit
def _group_first_scan_operator_alias_kernel(
    d_input,
    d_common_block_plus,
    d_common_block_sum,
    d_common_block_multiply,
    d_qualified_block_plus,
    d_qualified_block_sum,
    d_qualified_block_multiply,
    d_common_warp_plus,
    d_common_warp_sum,
    d_common_warp_multiply,
    d_qualified_warp_plus,
    d_qualified_warp_sum,
    d_qualified_warp_multiply,
):
    tid = cuda.threadIdx.x
    value = d_input[tid]
    common_block = coop.this_block()
    qualified_block = numba_coop.this_block()
    common_warp = coop.this_warp()
    qualified_warp = numba_coop.this_warp()

    d_common_block_plus[tid] = coop.scan(
        common_block,
        value,
        mode="exclusive",
        scan_op="+",
    )
    d_common_block_sum[tid] = coop.exclusive_scan(
        common_block,
        value,
        scan_op="sum",
    )
    d_common_block_multiply[tid] = coop.exclusive_scan(
        common_block,
        value,
        scan_op="multiply",
        initial_value=1,
    )
    d_qualified_block_plus[tid] = numba_coop.scan(
        qualified_block,
        value,
        mode="exclusive",
        scan_op="+",
    )
    d_qualified_block_sum[tid] = numba_coop.exclusive_scan(
        qualified_block,
        value,
        scan_op="sum",
    )
    d_qualified_block_multiply[tid] = numba_coop.exclusive_scan(
        qualified_block,
        value,
        scan_op="multiply",
        initial_value=1,
    )

    d_common_warp_plus[tid] = coop.scan(
        common_warp,
        value,
        mode="exclusive",
        scan_op="+",
    )
    d_common_warp_sum[tid] = coop.exclusive_scan(
        common_warp,
        value,
        scan_op="sum",
    )
    d_common_warp_multiply[tid] = coop.exclusive_scan(
        common_warp,
        value,
        scan_op="multiply",
        initial_value=1,
    )
    d_qualified_warp_plus[tid] = numba_coop.scan(
        qualified_warp,
        value,
        mode="exclusive",
        scan_op="+",
    )
    d_qualified_warp_sum[tid] = numba_coop.exclusive_scan(
        qualified_warp,
        value,
        scan_op="sum",
    )
    d_qualified_warp_multiply[tid] = numba_coop.exclusive_scan(
        qualified_warp,
        value,
        scan_op="multiply",
        initial_value=1,
    )


@pytest.mark.evidence_for(
    "group.exclusive_scan", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for("group.scan", backend="numba_mlir", evidence="runtime")
def test_group_first_scan_operator_aliases_are_defined_for_block_and_warp_roots(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    threads = 2 * _THREADS
    values = np.ones(threads, dtype=np.int32)
    values[[0, 31, 32, 63]] = np.asarray([2, 3, 5, 7], dtype=np.int32)
    outputs = [np.zeros_like(values) for _ in range(12)]

    _group_first_scan_operator_alias_kernel[1, threads](values, *outputs)
    cuda.synchronize()

    expected_block_sum = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(values[:-1], dtype=np.int32))
    )
    expected_block_multiply = np.concatenate(
        (
            np.ones(1, dtype=np.int32),
            np.cumprod(values[:-1], dtype=np.int32),
        )
    )
    expected_warp_sum = np.empty_like(values)
    expected_warp_multiply = np.empty_like(values)
    for warp in range(2):
        begin = warp * _THREADS
        end = begin + _THREADS
        warp_values = values[begin:end]
        expected_warp_sum[begin] = 0
        expected_warp_sum[begin + 1 : end] = np.cumsum(warp_values[:-1], dtype=np.int32)
        expected_warp_multiply[begin] = 1
        expected_warp_multiply[begin + 1 : end] = np.cumprod(
            warp_values[:-1], dtype=np.int32
        )

    for actual in (*outputs[0:2], *outputs[3:5]):
        np.testing.assert_array_equal(actual, expected_block_sum)
        assert actual[0] == 0
    for actual in (outputs[2], outputs[5]):
        np.testing.assert_array_equal(actual, expected_block_multiply)
    for actual in (*outputs[6:8], *outputs[9:11]):
        np.testing.assert_array_equal(actual, expected_warp_sum)
        np.testing.assert_array_equal(actual[::_THREADS], np.zeros(2, dtype=np.int32))
    for actual in (outputs[8], outputs[11]):
        np.testing.assert_array_equal(actual, expected_warp_multiply)


@cuda.jit
def _common_root_max_kernel(d_input, d_block_max, d_warp_max, d_original):
    tid = cuda.threadIdx.x
    value = d_input[tid]
    block_max = coop.reduce(
        coop.this_block(),
        value,
        binary_op="max",
        broadcast=False,
        algorithm="raking",
    )
    warp_max = coop.reduce(
        coop.this_warp(),
        value,
        binary_op="max",
        broadcast=False,
        valid_items=_VALID_WARP_ITEMS,
    )
    if tid == 0:
        d_block_max[0] = block_max
    if tid % _THREADS == 0:
        d_warp_max[tid // _THREADS] = warp_max
    d_original[tid] = value


@cuda.jit
def _qualified_root_max_kernel(d_input, d_block_max, d_warp_max, d_original):
    tid = cuda.threadIdx.x
    value = d_input[tid]
    block_max = numba_coop.reduce(
        numba_coop.this_block(),
        value,
        binary_op="max",
        broadcast=False,
        algorithm="raking",
    )
    warp_max = numba_coop.reduce(
        numba_coop.this_warp(),
        value,
        binary_op="max",
        broadcast=False,
        valid_items=_VALID_WARP_ITEMS,
    )
    if tid == 0:
        d_block_max[0] = block_max
    if tid % _THREADS == 0:
        d_warp_max[tid // _THREADS] = warp_max
    d_original[tid] = value


@pytest.mark.evidence_for("group.reduce", backend="numba_mlir", evidence="runtime")
def test_common_root_max_matches_qualified_numba_and_independent_oracle(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    warps = 2
    threads = warps * _THREADS
    values = ((np.arange(threads, dtype=np.int32) * 17) % 53) - 26
    values_before = values.copy()
    common_outputs = (
        np.full(1, -999, dtype=np.int32),
        np.full(warps, -999, dtype=np.int32),
        np.zeros_like(values),
    )
    qualified_outputs = tuple(np.empty_like(output) for output in common_outputs)
    for output, common_output in zip(
        qualified_outputs,
        common_outputs,
        strict=True,
    ):
        output[...] = common_output

    _common_root_max_kernel[1, threads](values, *common_outputs)
    _qualified_root_max_kernel[1, threads](values, *qualified_outputs)
    cuda.synchronize()

    for common_output, qualified_output in zip(
        common_outputs,
        qualified_outputs,
        strict=True,
    ):
        assert common_output.dtype == np.dtype(np.int32)
        np.testing.assert_array_equal(common_output, qualified_output)

    expected_warp_max = np.empty(warps, dtype=np.int32)
    for warp in range(warps):
        begin = warp * _THREADS
        expected_warp_max[warp] = values[begin : begin + _VALID_WARP_ITEMS].max()
    np.testing.assert_array_equal(common_outputs[0], np.asarray([values.max()]))
    np.testing.assert_array_equal(common_outputs[1], expected_warp_max)
    np.testing.assert_array_equal(common_outputs[2], values_before)
    np.testing.assert_array_equal(values, values_before)
