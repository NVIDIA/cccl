# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Differential GPU evidence for portable keys-only block TopK."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _partial_topk_kernel(
    source,
    common_max,
    qualified_max,
    common_min,
    qualified_min,
    common_original,
    qualified_original,
    k,
    valid_items,
    begin_bit,
    end_bit,
):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_keys = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=source.dtype,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        value = source[index]
        common_keys[item] = value
        qualified_keys[item] = value

    common_max_result = coop.topk_max_keys(
        coop.this_block(),
        common_keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    qualified_max_result = numba_coop.topk_max_keys(
        numba_coop.this_block(),
        qualified_keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
    )
    common_min_result = coop.topk_min_keys(
        coop.this_block(),
        common_keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
        end_bit=end_bit,
    )
    qualified_min_result = numba_coop.topk_min_keys(
        numba_coop.this_block(),
        qualified_keys,
        k,
        valid_items=valid_items,
        begin_bit=begin_bit,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        if index < k:
            common_max[index] = common_max_result[item]
            qualified_max[index] = qualified_max_result[item]
            common_min[index] = common_min_result[item]
            qualified_min[index] = qualified_min_result[item]
        common_original[index] = common_keys[item]
        qualified_original[index] = qualified_keys[item]


@cuda.jit
def _full_topk_kernel(
    source,
    common_max,
    qualified_max,
    common_min,
    qualified_min,
    k,
):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=source.dtype)
    qualified_keys = numba_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=source.dtype,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        value = source[index]
        common_keys[item] = value
        qualified_keys[item] = value

    common_max_result = coop.topk_max_keys(coop.this_block(), common_keys, k)
    qualified_max_result = numba_coop.topk_max_keys(
        numba_coop.this_block(),
        qualified_keys,
        k,
    )
    common_min_result = coop.topk_min_keys(coop.this_block(), common_keys, k)
    qualified_min_result = numba_coop.topk_min_keys(
        numba_coop.this_block(),
        qualified_keys,
        k,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        if index < k:
            common_max[index] = common_max_result[item]
            qualified_max[index] = qualified_max_result[item]
            common_min[index] = common_min_result[item]
            qualified_min[index] = qualified_min_result[item]


@cuda.jit
def _pair_topk_kernel(
    keys_source,
    values_source,
    common_max_keys,
    common_max_values,
    qualified_max_keys,
    qualified_max_values,
    common_min_keys,
    common_min_values,
    qualified_min_keys,
    qualified_min_values,
    original_keys,
    original_values,
    k,
    valid_items,
):
    tid = cuda.threadIdx.x
    common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=keys_source.dtype)
    common_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=values_source.dtype)
    qualified_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=keys_source.dtype)
    qualified_values = numba_coop.ThreadData(
        _ITEMS_PER_THREAD, dtype=values_source.dtype
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        common_keys[item] = keys_source[index]
        common_values[item] = values_source[index]
        qualified_keys[item] = keys_source[index]
        qualified_values[item] = values_source[index]

    common_max = coop.topk_max_pairs(coop.this_block(), common_keys, common_values, k)
    qualified_max = numba_coop.topk_max_pairs(
        numba_coop.this_block(), qualified_keys, qualified_values, k
    )
    common_min = coop.topk_min_pairs(
        coop.this_block(),
        common_keys,
        common_values,
        k,
        valid_items=valid_items,
    )
    qualified_min = numba_coop.topk_min_pairs(
        numba_coop.this_block(),
        qualified_keys,
        qualified_values,
        k,
        valid_items=valid_items,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        if index < k:
            common_max_keys[index] = common_max[0][item]
            common_max_values[index] = common_max[1][item]
            qualified_max_keys[index] = qualified_max[0][item]
            qualified_max_values[index] = qualified_max[1][item]
            common_min_keys[index] = common_min[0][item]
            common_min_values[index] = common_min[1][item]
            qualified_min_keys[index] = qualified_min[0][item]
            qualified_min_values[index] = qualified_min[1][item]
        original_keys[index] = common_keys[item]
        original_values[index] = common_values[item]


def _keys(dtype):
    resolved = np.dtype(dtype)
    bit_width = resolved.itemsize * 8
    mask = (1 << bit_width) - 1
    multiplier = 0x9E37_79B9_7F4A_7C15 & mask
    raw = np.asarray(
        [
            ((index * multiplier) ^ (index << 11) ^ (index * index * 29)) & mask
            for index in range(_TOTAL_ITEMS)
        ],
        dtype=np.dtype(f"uint{bit_width}"),
    )
    raw[9::13] = raw[8::13]
    if np.issubdtype(resolved, np.signedinteger):
        return raw.view(resolved).copy()
    return raw.astype(resolved, copy=False)


def _ordered_digits(values, begin_bit, end_bit):
    bit_width = values.dtype.itemsize * 8
    unsigned = values.view(np.dtype(f"uint{bit_width}")).copy()
    if np.issubdtype(values.dtype, np.signedinteger):
        unsigned ^= np.asarray(1 << (bit_width - 1), dtype=unsigned.dtype)
    digit_mask = (1 << (end_bit - begin_bit)) - 1
    return (unsigned >> begin_bit) & np.asarray(digit_mask, dtype=unsigned.dtype)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.evidence_for(
    "group.topk_max_keys",
    backend="numba_mlir",
    evidence="runtime",
)
@pytest.mark.evidence_for(
    "group.topk_min_keys",
    backend="numba_mlir",
    evidence="runtime",
)
def test_common_and_qualified_partial_topk_match_bit_ordered_oracle_and_preserve_input(
    dtype,
):
    values = _keys(dtype)
    valid_items = _TOTAL_ITEMS - 7
    k = 11
    bit_width = values.dtype.itemsize * 8
    begin_bit = bit_width - 5
    outputs = [np.zeros_like(values) for _ in range(4)]
    originals = [np.zeros_like(values) for _ in range(2)]

    _partial_topk_kernel[1, _THREADS](
        values,
        *outputs,
        *originals,
        np.int32(k),
        np.int32(valid_items),
        np.int32(begin_bit),
        np.int32(bit_width),
    )

    common_max, qualified_max, common_min, qualified_min = outputs
    np.testing.assert_array_equal(common_max[:k], qualified_max[:k])
    np.testing.assert_array_equal(common_min[:k], qualified_min[:k])
    np.testing.assert_array_equal(originals[0], values)
    np.testing.assert_array_equal(originals[1], values)

    valid_digits = _ordered_digits(values[:valid_items], begin_bit, bit_width)
    max_digits = _ordered_digits(common_max[:k], begin_bit, bit_width)
    min_digits = _ordered_digits(common_min[:k], begin_bit, bit_width)
    np.testing.assert_array_equal(np.sort(max_digits), np.sort(valid_digits)[-k:])
    np.testing.assert_array_equal(np.sort(min_digits), np.sort(valid_digits)[:k])


@pytest.mark.parametrize("k", [1, _TOTAL_ITEMS])
@pytest.mark.evidence_for(
    "group.topk_max_keys",
    backend="numba_mlir",
    evidence="runtime",
)
@pytest.mark.evidence_for(
    "group.topk_min_keys",
    backend="numba_mlir",
    evidence="runtime",
)
def test_common_and_qualified_full_topk_cover_k_boundaries(k):
    values = _keys(np.int32)
    outputs = [np.zeros_like(values) for _ in range(4)]

    _full_topk_kernel[1, _THREADS](values, *outputs, np.int32(k))

    common_max, qualified_max, common_min, qualified_min = outputs
    np.testing.assert_array_equal(common_max[:k], qualified_max[:k])
    np.testing.assert_array_equal(common_min[:k], qualified_min[:k])
    np.testing.assert_array_equal(np.sort(common_max[:k]), np.sort(values)[-k:])
    np.testing.assert_array_equal(np.sort(common_min[:k]), np.sort(values)[:k])


@pytest.mark.evidence_for(
    "group.topk_max_pairs", backend="numba_mlir", evidence="runtime"
)
@pytest.mark.evidence_for(
    "group.topk_min_pairs", backend="numba_mlir", evidence="runtime"
)
def test_common_topk_pairs_match_qualified_oracles_and_preserve_association():
    keys = _keys(np.int32)
    values = np.arange(_TOTAL_ITEMS, dtype=np.float64) + 0.5
    k = 11
    valid_items = _TOTAL_ITEMS - 7
    key_outputs = [np.zeros_like(keys) for _ in range(5)]
    value_outputs = [np.zeros_like(values) for _ in range(5)]
    _pair_topk_kernel[1, _THREADS](
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
        np.int32(k),
        np.int32(valid_items),
    )

    np.testing.assert_array_equal(key_outputs[4], keys)
    np.testing.assert_array_equal(value_outputs[4], values)
    for common_index, qualified_index in ((0, 1), (2, 3)):
        np.testing.assert_array_equal(
            np.sort(key_outputs[common_index][:k]),
            np.sort(key_outputs[qualified_index][:k]),
        )
        assert set(
            zip(key_outputs[common_index][:k], value_outputs[common_index][:k])
        ) <= set(
            zip(
                keys[:valid_items] if common_index == 2 else keys,
                values[:valid_items] if common_index == 2 else values,
            )
        )
    np.testing.assert_array_equal(np.sort(key_outputs[0][:k]), np.sort(keys)[-k:])
    np.testing.assert_array_equal(
        np.sort(key_outputs[2][:k]), np.sort(keys[:valid_items])[:k]
    )
