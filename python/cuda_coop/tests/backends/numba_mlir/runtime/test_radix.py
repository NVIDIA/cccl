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

import cuda.coop.numba_mlir as coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_THREADS = 64
_ITEMS_PER_THREAD = 2
_TOTAL_ITEMS = _THREADS * _ITEMS_PER_THREAD
_BEGIN_BIT = 4
_END_BIT = 12
_RANK_BITS = 4


@cuda.jit
def _radix_kernel(
    source_keys,
    source_values,
    original_keys,
    sorted_pair_keys,
    sorted_pair_values,
    descending_keys,
    ranks_out,
    prefixes_out,
):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = source_keys[index]
        values[item] = source_values[index]

    fixed_storage = coop.TempStorage(8192, alignment=16)
    pair_keys, pair_values = coop.radix_sort_pairs(
        coop.this_block(),
        keys,
        values,
        begin_bit=_BEGIN_BIT,
        end_bit=_END_BIT,
        temp_storage=fixed_storage,
    )
    deferred_storage = coop.TempStorage()
    descending = coop.radix_sort_keys(
        coop.this_block(),
        keys,
        descending=True,
        temp_storage=deferred_storage,
    )
    prefix = coop.ThreadData(1, dtype=types.int32)
    ranks = coop.radix_rank(
        coop.this_block(),
        keys,
        begin_bit=0,
        radix_bits=_RANK_BITS,
        exclusive_digit_prefix=prefix,
    )

    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        original_keys[index] = keys[item]
        sorted_pair_keys[index] = pair_keys[item]
        sorted_pair_values[index] = pair_values[item]
        descending_keys[index] = descending[item]
        ranks_out[index] = ranks[item]
    prefixes_out[tid] = prefix[0]


def _rank_reference(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    digits = keys.view(np.uint32) & np.uint32((1 << _RANK_BITS) - 1)
    counts = np.bincount(digits, minlength=1 << _RANK_BITS)
    prefixes = np.concatenate(([0], np.cumsum(counts[:-1]))).astype(np.int32)
    seen = np.zeros(1 << _RANK_BITS, dtype=np.int32)
    ranks = np.empty(keys.size, dtype=np.int32)
    for index, digit in enumerate(digits):
        digit_index = int(digit)
        ranks[index] = prefixes[digit_index] + seen[digit_index]
        seen[digit_index] += 1
    return ranks, prefixes


def test_keys_pairs_runtime_bits_rank_prefix_and_fresh_results() -> None:
    indices = np.arange(_TOTAL_ITEMS, dtype=np.int32)
    keys = ((indices * np.int32(53)) % np.int32(257)) - np.int32(128)
    values = indices * np.int32(17) + np.int32(3)
    original = np.full_like(keys, -9999)
    pair_keys = np.full_like(keys, -9999)
    pair_values = np.full_like(values, -9999)
    descending = np.full_like(keys, -9999)
    ranks = np.full_like(keys, -9999)
    prefixes = np.full(_THREADS, -9999, dtype=np.int32)

    _radix_kernel[1, _THREADS](
        keys,
        values,
        original,
        pair_keys,
        pair_values,
        descending,
        ranks,
        prefixes,
    )

    np.testing.assert_array_equal(original, keys)
    selected_digits = (keys.view(np.uint32) >> np.uint32(_BEGIN_BIT)) & np.uint32(
        (1 << (_END_BIT - _BEGIN_BIT)) - 1
    )
    pair_order = np.argsort(selected_digits, kind="stable")
    np.testing.assert_array_equal(pair_keys, keys[pair_order])
    np.testing.assert_array_equal(pair_values, values[pair_order])
    np.testing.assert_array_equal(descending, np.sort(keys)[::-1])

    expected_ranks, expected_prefixes = _rank_reference(keys)
    np.testing.assert_array_equal(ranks, expected_ranks)
    np.testing.assert_array_equal(prefixes[: 1 << _RANK_BITS], expected_prefixes)
    np.testing.assert_array_equal(keys, original)
