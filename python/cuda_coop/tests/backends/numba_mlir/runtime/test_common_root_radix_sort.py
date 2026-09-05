# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Differential GPU evidence for common keys-only BlockRadixSort."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD
_BEGIN_ONLY = 7
_SUBRANGE_BEGIN = 4
_SUBRANGE_END = 13

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
)


@cuda.jit
def _radix_sort_kernel(
    source,
    common_original,
    qualified_original,
    common_full,
    qualified_full,
    common_begin_only,
    qualified_begin_only,
    common_subrange,
    qualified_subrange,
    begin_only,
    subrange_begin,
    subrange_end,
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

    common_group = coop.this_block()
    qualified_group = numba_coop.this_block()
    common_storage = coop.TempStorage()
    qualified_storage = numba_coop.TempStorage()
    common_full_result = coop.radix_sort_keys(
        common_group,
        common_keys,
        temp_storage=common_storage,
    )
    qualified_full_result = numba_coop.radix_sort_keys(
        qualified_group,
        qualified_keys,
        temp_storage=qualified_storage,
    )
    common_begin_only_result = coop.radix_sort_keys(
        common_group,
        common_keys,
        begin_bit=begin_only,
        descending=True,
    )
    qualified_begin_only_result = numba_coop.radix_sort_keys(
        qualified_group,
        qualified_keys,
        begin_bit=begin_only,
        descending=True,
    )
    common_subrange_result = coop.radix_sort_keys(
        common_group,
        common_keys,
        begin_bit=subrange_begin,
        end_bit=subrange_end,
    )
    qualified_subrange_result = numba_coop.radix_sort_keys(
        qualified_group,
        qualified_keys,
        begin_bit=subrange_begin,
        end_bit=subrange_end,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        common_original[index] = common_keys[item]
        qualified_original[index] = qualified_keys[item]
        common_full[index] = common_full_result[item]
        qualified_full[index] = qualified_full_result[item]
        common_begin_only[index] = common_begin_only_result[item]
        qualified_begin_only[index] = qualified_begin_only_result[item]
        common_subrange[index] = common_subrange_result[item]
        qualified_subrange[index] = qualified_subrange_result[item]


@cuda.jit
def _radix_sort_pairs_kernel(
    keys_source,
    values_source,
    common_keys_output,
    common_values_output,
    qualified_keys_output,
    qualified_values_output,
    original_keys,
    original_values,
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

    common_result = coop.radix_sort_pairs(
        coop.this_block(), common_keys, common_values, descending=True
    )
    qualified_result = numba_coop.radix_sort_pairs(
        numba_coop.this_block(), qualified_keys, qualified_values, descending=True
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        common_keys_output[index] = common_result[0][item]
        common_values_output[index] = common_result[1][item]
        qualified_keys_output[index] = qualified_result[0][item]
        qualified_values_output[index] = qualified_result[1][item]
        original_keys[index] = common_keys[item]
        original_values[index] = common_values[item]


def _keys_with_high_bits(dtype: type[np.generic]) -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    bit_width = resolved_dtype.itemsize * 8
    mask = (1 << bit_width) - 1
    high_bit = 1 << (bit_width - 1)
    multiplier = 0x9E37_79B9_7F4A_7C15 & mask
    raw = [
        ((index * multiplier) ^ (index << 17) ^ (index * index * 29)) & mask
        for index in range(_TOTAL_ITEMS)
    ]
    for index in range(11, _TOTAL_ITEMS, 17):
        raw[index] = raw[index - 1]
    raw[:6] = [0, high_bit, mask, high_bit - 1, 1, high_bit | 1]

    unsigned_dtype = np.dtype(f"uint{bit_width}")
    unsigned = np.asarray(raw, dtype=unsigned_dtype)
    if np.issubdtype(resolved_dtype, np.signedinteger):
        return unsigned.view(resolved_dtype).copy()
    return unsigned.astype(resolved_dtype, copy=False)


def _cub_bit_order(
    values: np.ndarray,
    *,
    begin_bit: int,
    end_bit: int,
    descending: bool,
) -> np.ndarray:
    """Apply CUB's documented integral-key transform and stable digit order."""

    bit_width = values.dtype.itemsize * 8
    raw_mask = (1 << bit_width) - 1
    sign_bit = 1 << (bit_width - 1)
    digit_mask = (1 << (end_bit - begin_bit)) - 1
    signed = np.issubdtype(values.dtype, np.signedinteger)

    def digit(index: int) -> int:
        ordered = int(values[index]) & raw_mask
        if signed:
            ordered ^= sign_bit
        return (ordered >> begin_bit) & digit_mask

    order = sorted(
        range(values.size),
        key=lambda index: ((-digit(index) if descending else digit(index)), index),
    )
    return values[np.asarray(order, dtype=np.intp)]


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="numba_mlir", evidence="runtime"
)
def test_common_and_qualified_radix_sort_match_cub_bit_order_and_preserve_input(
    dtype,
):
    values = _keys_with_high_bits(dtype)
    outputs = [np.full_like(values, 7) for _ in range(8)]

    _radix_sort_kernel[1, _THREADS](
        values,
        *outputs,
        np.int32(_BEGIN_ONLY),
        np.int32(_SUBRANGE_BEGIN),
        np.int32(_SUBRANGE_END),
    )

    (
        common_original,
        qualified_original,
        common_full,
        qualified_full,
        common_begin_only,
        qualified_begin_only,
        common_subrange,
        qualified_subrange,
    ) = outputs
    np.testing.assert_array_equal(common_original, values)
    np.testing.assert_array_equal(qualified_original, values)

    bit_width = values.dtype.itemsize * 8
    for common_result, qualified_result, begin_bit, end_bit, descending in (
        (common_full, qualified_full, 0, bit_width, False),
        (
            common_begin_only,
            qualified_begin_only,
            _BEGIN_ONLY,
            bit_width,
            True,
        ),
        (
            common_subrange,
            qualified_subrange,
            _SUBRANGE_BEGIN,
            _SUBRANGE_END,
            False,
        ),
    ):
        np.testing.assert_array_equal(common_result, qualified_result)
        np.testing.assert_array_equal(
            common_result,
            _cub_bit_order(
                values,
                begin_bit=begin_bit,
                end_bit=end_bit,
                descending=descending,
            ),
        )


@pytest.mark.evidence_for(
    "group.radix_sort_pairs", backend="numba_mlir", evidence="runtime"
)
def test_common_radix_sort_pairs_match_qualified_and_preserve_association():
    keys = _keys_with_high_bits(np.int64)
    values = np.arange(_TOTAL_ITEMS, dtype=np.float32) + np.float32(0.25)
    key_outputs = [np.zeros_like(keys) for _ in range(3)]
    value_outputs = [np.zeros_like(values) for _ in range(3)]
    _radix_sort_pairs_kernel[1, _THREADS](
        keys,
        values,
        key_outputs[0],
        value_outputs[0],
        key_outputs[1],
        value_outputs[1],
        key_outputs[2],
        value_outputs[2],
    )

    np.testing.assert_array_equal(key_outputs[2], keys)
    np.testing.assert_array_equal(value_outputs[2], values)
    np.testing.assert_array_equal(key_outputs[0], key_outputs[1])
    np.testing.assert_array_equal(value_outputs[0], value_outputs[1])
    np.testing.assert_array_equal(
        key_outputs[0], _cub_bit_order(keys, begin_bit=0, end_bit=64, descending=True)
    )
    assert set(zip(key_outputs[0], value_outputs[0])) == set(zip(keys, values))
