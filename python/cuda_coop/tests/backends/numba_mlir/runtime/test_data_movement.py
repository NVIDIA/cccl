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

from cuda import coop

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
_WIDE_ITEMS_PER_THREAD = 8
_LOGICAL_WARP_THREADS = 8
_STATIC_OFFSET = 4
_STATIC_VALID_ITEMS = _THREADS * _ITEMS_PER_THREAD - 8


@cuda.jit
def _block_load_store_fixed_storage(source, output):
    storage = coop.TempStorage(4096, alignment=16)
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded = coop.load(
        coop.this_block(),
        source,
        items,
        algorithm="transpose",
        temp_storage=storage,
    )
    coop.store(
        coop.this_block(),
        output,
        loaded,
        algorithm="transpose",
        temp_storage=storage,
    )


def test_block_load_store_forwards_fixed_capacity_temp_storage():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _block_load_store_fixed_storage[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)


@cuda.jit
def _block_load_static_controls(source, output):
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded = coop.load(
        coop.this_block(),
        source,
        items,
        valid_items=_STATIC_VALID_ITEMS,
        oob_default=-1,
        offset=_STATIC_OFFSET,
    )
    coop.store(coop.this_block(), output, loaded)


def test_block_load_static_controls_match_independent_oracle():
    tile_items = _THREADS * _ITEMS_PER_THREAD
    source = np.arange(tile_items + _STATIC_OFFSET, dtype=np.int32)
    output = np.full(tile_items, -7, dtype=np.int32)
    expected = np.full(tile_items, -1, dtype=np.int32)
    expected[:_STATIC_VALID_ITEMS] = source[
        _STATIC_OFFSET : _STATIC_OFFSET + _STATIC_VALID_ITEMS
    ]

    _block_load_static_controls[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, expected)


@cuda.jit
def _logical_warp_load_store(source, output):
    logical_warp = coop.this_warp().group_by(_LOGICAL_WARP_THREADS)
    items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    loaded = coop.load(logical_warp, source, items)
    coop.store(logical_warp, output, loaded)


def test_logical_width_8_load_store_round_trip():
    source = np.arange(_THREADS * _ITEMS_PER_THREAD, dtype=np.int32)
    output = np.full_like(source, -1)

    _logical_warp_load_store[1, _THREADS](source, output)

    np.testing.assert_array_equal(output, source)
