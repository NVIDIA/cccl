# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 32
_ITEMS_PER_THREAD = 2
_TILE_ITEMS = _THREADS * _ITEMS_PER_THREAD


@cuda.jit
def _topk_kernel(
    key_source,
    value_source,
    max_keys,
    min_keys,
    min_values,
    original_keys,
    original_values,
    k,
    valid_items,
):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    values = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = key_source[index]
        values[item] = value_source[index]

    selected_max = coop.topk_max_keys(
        coop.this_block(),
        keys,
        k,
        valid_items=valid_items,
        begin_bit=1,
        end_bit=32,
    )
    selected_min_keys, selected_min_values = numba_coop.topk_min_pairs(
        numba_coop.this_block(),
        keys,
        values,
        k,
        valid_items=valid_items,
        begin_bit=1,
    )
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        if index < k:
            max_keys[index] = selected_max[item]
            min_keys[index] = selected_min_keys[item]
            min_values[index] = selected_min_values[item]
        original_keys[index] = keys[item]
        original_values[index] = values[item]


def _ordered_digits(values):
    unsigned = values.view(np.uint32).copy()
    unsigned ^= np.uint32(1 << 31)
    return unsigned >> np.uint32(1)


def test_topk_keys_and_pairs_match_oracles_and_preserve_inputs() -> None:
    indices = np.arange(_TILE_ITEMS, dtype=np.int32)
    keys = ((indices * 1_103_515_245 + 12_345) ^ (indices << 7)).astype(np.int32)
    values = np.arange(_TILE_ITEMS, dtype=np.float32) + np.float32(0.5)
    k = 11
    valid_items = _TILE_ITEMS - 7
    max_keys = np.zeros_like(keys)
    min_keys = np.zeros_like(keys)
    min_values = np.zeros_like(values)
    original_keys = np.zeros_like(keys)
    original_values = np.zeros_like(values)

    _topk_kernel[1, _THREADS](
        keys,
        values,
        max_keys,
        min_keys,
        min_values,
        original_keys,
        original_values,
        np.int32(k),
        np.int32(valid_items),
    )

    np.testing.assert_array_equal(original_keys, keys)
    np.testing.assert_array_equal(original_values, values)
    valid_digits = _ordered_digits(keys[:valid_items])
    np.testing.assert_array_equal(
        np.sort(_ordered_digits(max_keys[:k])),
        np.sort(valid_digits)[-k:],
    )
    np.testing.assert_array_equal(
        np.sort(_ordered_digits(min_keys[:k])),
        np.sort(valid_digits)[:k],
    )
    assert set(zip(min_keys[:k], min_values[:k])) <= set(
        zip(keys[:valid_items], values[:valid_items])
    )
