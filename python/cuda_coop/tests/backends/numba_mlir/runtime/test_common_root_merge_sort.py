# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Differential GPU evidence for common keys-only MergeSort."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD
_WARP_ITEMS = 32 * _ITEMS_PER_THREAD
_BLOCK_VALID_ITEMS = 117
_WARP_VALID_ITEMS = 53
_BLOCK_DESCENDING_OOB_DEFAULT = -2_147_483_648
_WARP_ASCENDING_OOB_DEFAULT = 2_147_483_647

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit(device=True)
def _less(lhs, rhs):
    return lhs < rhs


@cuda.jit(device=True)
def _greater(lhs, rhs):
    return lhs > rhs


@cuda.jit
def _block_kernel(
    source, original, common_full, qualified_full, common_partial, qualified_partial
):
    tid = cuda.threadIdx.x
    common_full_keys = coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_full_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    common_partial_keys = coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_partial_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        value = source[tid * _ITEMS_PER_THREAD + item]
        common_full_keys[item] = value
        qualified_full_keys[item] = value
        common_partial_keys[item] = value
        qualified_partial_keys[item] = value

    common_group = coop.this_block()
    qualified_group = numba_coop.this_block()
    common_full_result = coop.merge_sort_keys(common_group, common_full_keys)
    qualified_full_result = numba_coop.merge_sort_keys(
        qualified_group,
        qualified_full_keys,
        compare_op=_less,
    )
    common_partial_result = coop.merge_sort_keys(
        common_group,
        common_partial_keys,
        descending=True,
        valid_items=_BLOCK_VALID_ITEMS,
        oob_default=_BLOCK_DESCENDING_OOB_DEFAULT,
    )
    qualified_partial_result = numba_coop.merge_sort_keys(
        qualified_group,
        qualified_partial_keys,
        descending=True,
        valid_items=_BLOCK_VALID_ITEMS,
        oob_default=_BLOCK_DESCENDING_OOB_DEFAULT,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original[index] = common_full_keys[item]
        common_full[index] = common_full_result[item]
        qualified_full[index] = qualified_full_result[item]
        common_partial[index] = common_partial_result[item]
        qualified_partial[index] = qualified_partial_result[item]


@cuda.jit
def _warp_kernel(
    source, original, common_full, qualified_full, common_partial, qualified_partial
):
    tid = cuda.threadIdx.x
    common_full_keys = coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_full_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    common_partial_keys = coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_partial_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        value = source[tid * _ITEMS_PER_THREAD + item]
        common_full_keys[item] = value
        qualified_full_keys[item] = value
        common_partial_keys[item] = value
        qualified_partial_keys[item] = value

    common_group = coop.this_warp()
    qualified_group = numba_coop.this_warp()
    common_full_result = coop.merge_sort_keys(
        common_group,
        common_full_keys,
        descending=True,
    )
    qualified_full_result = numba_coop.merge_sort_keys(
        qualified_group,
        qualified_full_keys,
        compare_op=_greater,
    )
    common_partial_result = coop.merge_sort_keys(
        common_group,
        common_partial_keys,
        valid_items=_WARP_VALID_ITEMS,
        oob_default=_WARP_ASCENDING_OOB_DEFAULT,
    )
    qualified_partial_result = numba_coop.merge_sort_keys(
        qualified_group,
        qualified_partial_keys,
        valid_items=_WARP_VALID_ITEMS,
        oob_default=_WARP_ASCENDING_OOB_DEFAULT,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original[index] = common_full_keys[item]
        common_full[index] = common_full_result[item]
        qualified_full[index] = qualified_full_result[item]
        common_partial[index] = common_partial_result[item]
        qualified_partial[index] = qualified_partial_result[item]


@cuda.jit
def _portable_dtype_kernel(
    source,
    original,
    common_block,
    qualified_block,
    common_warp,
    qualified_warp,
):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_keys = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=source.dtype,
    )
    for item in range(_ITEMS_PER_THREAD):
        value = source[tid * _ITEMS_PER_THREAD + item]
        common_keys[item] = value
        qualified_keys[item] = value

    common_block_result = coop.merge_sort_keys(
        coop.this_block(),
        common_keys,
        temp_storage=coop.TempStorage(),
    )
    qualified_block_result = numba_coop.merge_sort_keys(
        numba_coop.this_block(),
        qualified_keys,
        temp_storage=numba_coop.TempStorage(),
    )
    common_warp_result = coop.merge_sort_keys(
        coop.this_warp(),
        common_keys,
        descending=True,
    )
    qualified_warp_result = numba_coop.merge_sort_keys(
        numba_coop.this_warp(),
        qualified_keys,
        descending=True,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original[index] = common_keys[item]
        common_block[index] = common_block_result[item]
        qualified_block[index] = qualified_block_result[item]
        common_warp[index] = common_warp_result[item]
        qualified_warp[index] = qualified_warp_result[item]


@cuda.jit
def _pair_kernel(
    keys_source,
    values_source,
    common_block_keys,
    common_block_values,
    qualified_block_keys,
    qualified_block_values,
    common_warp_keys,
    common_warp_values,
    qualified_warp_keys,
    qualified_warp_values,
    original_keys,
    original_values,
):
    tid = cuda.threadIdx.x
    # Infer the key and value dtypes independently from their indexed writes.
    common_block_thread_keys = coop.ThreadData(_ITEMS_PER_THREAD)
    common_block_thread_values = coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_block_thread_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_block_thread_values = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    common_warp_thread_keys = coop.ThreadData(_ITEMS_PER_THREAD)
    common_warp_thread_values = coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_warp_thread_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    qualified_warp_thread_values = numba_coop.ThreadData(_ITEMS_PER_THREAD)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        key = keys_source[index]
        value = values_source[index]
        common_block_thread_keys[item] = key
        common_block_thread_values[item] = value
        qualified_block_thread_keys[item] = key
        qualified_block_thread_values[item] = value
        common_warp_thread_keys[item] = key
        common_warp_thread_values[item] = value
        qualified_warp_thread_keys[item] = key
        qualified_warp_thread_values[item] = value

    common_block_result = coop.merge_sort_pairs(
        coop.this_block(), common_block_thread_keys, common_block_thread_values
    )
    qualified_block_result = numba_coop.merge_sort_pairs(
        numba_coop.this_block(),
        qualified_block_thread_keys,
        qualified_block_thread_values,
    )
    common_warp_result = coop.merge_sort_pairs(
        coop.this_warp(),
        common_warp_thread_keys,
        common_warp_thread_values,
        descending=True,
    )
    qualified_warp_result = numba_coop.merge_sort_pairs(
        numba_coop.this_warp(),
        qualified_warp_thread_keys,
        qualified_warp_thread_values,
        descending=True,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original_keys[index] = common_block_thread_keys[item]
        original_values[index] = common_block_thread_values[item]
        common_block_keys[index] = common_block_result[0][item]
        common_block_values[index] = common_block_result[1][item]
        qualified_block_keys[index] = qualified_block_result[0][item]
        qualified_block_values[index] = qualified_block_result[1][item]
        common_warp_keys[index] = common_warp_result[0][item]
        common_warp_values[index] = common_warp_result[1][item]
        qualified_warp_keys[index] = qualified_warp_result[0][item]
        qualified_warp_values[index] = qualified_warp_result[1][item]


def _duplicate_keys() -> np.ndarray:
    indices = np.arange(_TOTAL_ITEMS, dtype=np.int32)
    return ((indices * np.int32(11) + indices // np.int32(3)) % np.int32(19)) - 9


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="numba_mlir", evidence="runtime"
)
def test_common_and_qualified_merge_sort_match_independent_oracles_and_preserve_input():
    values = _duplicate_keys()
    assert _BLOCK_DESCENDING_OOB_DEFAULT < values[:_BLOCK_VALID_ITEMS].min()
    assert all(
        _WARP_ASCENDING_OOB_DEFAULT > values[base : base + _WARP_VALID_ITEMS].max()
        for base in range(0, _TOTAL_ITEMS, _WARP_ITEMS)
    )

    block_outputs = [np.full_like(values, -777_777) for _ in range(5)]
    _block_kernel[1, _THREADS](values, *block_outputs)
    original, common_full, qualified_full, common_partial, qualified_partial = (
        block_outputs
    )
    np.testing.assert_array_equal(original, values)
    np.testing.assert_array_equal(common_full, qualified_full)
    np.testing.assert_array_equal(common_full, np.sort(values))
    np.testing.assert_array_equal(
        common_partial[:_BLOCK_VALID_ITEMS],
        qualified_partial[:_BLOCK_VALID_ITEMS],
    )
    np.testing.assert_array_equal(
        common_partial[:_BLOCK_VALID_ITEMS],
        np.sort(values[:_BLOCK_VALID_ITEMS])[::-1],
    )

    warp_outputs = [np.full_like(values, -777_777) for _ in range(5)]
    _warp_kernel[1, _THREADS](values, *warp_outputs)
    original, common_full, qualified_full, common_partial, qualified_partial = (
        warp_outputs
    )
    np.testing.assert_array_equal(original, values)
    for base in range(0, _TOTAL_ITEMS, _WARP_ITEMS):
        end = base + _WARP_ITEMS
        np.testing.assert_array_equal(common_full[base:end], qualified_full[base:end])
        np.testing.assert_array_equal(
            common_full[base:end], np.sort(values[base:end])[::-1]
        )
        valid_end = base + _WARP_VALID_ITEMS
        np.testing.assert_array_equal(
            common_partial[base:valid_end],
            qualified_partial[base:valid_end],
        )
        np.testing.assert_array_equal(
            common_partial[base:valid_end],
            np.sort(values[base:valid_end]),
        )


def _portable_dtype_keys(dtype: type[np.generic]) -> np.ndarray:
    info = np.iinfo(dtype)
    values: list[int] = []
    for index in range(_TOTAL_ITEMS):
        magnitude = (index * 37 + (index % 7) * 11) % 53
        if info.min < 0:
            value = -magnitude if index % 2 else magnitude
            if info.bits == 64:
                value += (1 << 35) if index % 3 == 0 else -(1 << 34)
        else:
            value = magnitude
            if index % 3 == 0:
                value += 1 << (info.bits - 1)
            elif info.bits == 64:
                value += (index % 5) << 36
        values.append(value)
    return np.asarray(values, dtype=dtype)


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
def test_portable_integer_dtypes_preserve_type_and_match_block_and_warp_oracles(
    dtype: type[np.generic],
) -> None:
    values = _portable_dtype_keys(dtype)
    outputs = [np.zeros_like(values) for _ in range(5)]

    _portable_dtype_kernel[1, _THREADS](values, *outputs)

    original, common_block, qualified_block, common_warp, qualified_warp = outputs
    assert original.dtype == values.dtype
    assert common_block.dtype == qualified_block.dtype == values.dtype
    assert common_warp.dtype == qualified_warp.dtype == values.dtype
    np.testing.assert_array_equal(original, values)
    np.testing.assert_array_equal(common_block, qualified_block)
    np.testing.assert_array_equal(common_block, np.sort(values))
    for base in range(0, _TOTAL_ITEMS, _WARP_ITEMS):
        end = base + _WARP_ITEMS
        np.testing.assert_array_equal(common_warp[base:end], qualified_warp[base:end])
        np.testing.assert_array_equal(
            common_warp[base:end],
            np.sort(values[base:end])[::-1],
        )


@pytest.mark.evidence_for(
    "group.merge_sort_pairs", backend="numba_mlir", evidence="runtime"
)
def test_common_merge_sort_pairs_match_qualified_and_preserve_association():
    keys = _duplicate_keys()
    values = np.arange(_TOTAL_ITEMS, dtype=np.float64) + 0.5
    key_outputs = [np.zeros_like(keys) for _ in range(5)]
    value_outputs = [np.zeros_like(values) for _ in range(5)]
    _pair_kernel[1, _THREADS](
        keys,
        values,
        key_outputs[0],
        value_outputs[0],
        key_outputs[1],
        value_outputs[1],
        key_outputs[2],
        value_outputs[2],
        key_outputs[3],
        value_outputs[3],
        key_outputs[4],
        value_outputs[4],
    )

    np.testing.assert_array_equal(key_outputs[4], keys)
    np.testing.assert_array_equal(value_outputs[4], values)
    for common_index, qualified_index in ((0, 1), (2, 3)):
        np.testing.assert_array_equal(
            key_outputs[common_index], key_outputs[qualified_index]
        )
        np.testing.assert_array_equal(
            value_outputs[common_index], value_outputs[qualified_index]
        )
        assert set(zip(key_outputs[common_index], value_outputs[common_index])) == set(
            zip(keys, values)
        )
    np.testing.assert_array_equal(key_outputs[0], np.sort(keys))
    for base in range(0, _TOTAL_ITEMS, _WARP_ITEMS):
        end = base + _WARP_ITEMS
        np.testing.assert_array_equal(
            key_outputs[2][base:end], np.sort(keys[base:end])[::-1]
        )
